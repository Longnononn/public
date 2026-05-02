#!/usr/bin/env python3
"""
Repo Sync Utility for Nexus Infinite Dual-Repo Architecture
Syncs weights, datasets, and artifacts between Public and Private repos.

Usage:
    # Public -> Private (trigger training cycle)
    python scripts/repo_sync.py --direction public-to-private \\
        --artifact-url <url> --token $PRIVATE_CORE_TOKEN \\
        --target-repo owner/nexus-infinite-core
    
    # Private -> Public (sync best weights back)
    python scripts/repo_sync.py --direction private-to-public \\
        --weights-file models/best.nnue --token $PUBLIC_REPO_TOKEN \\
        --target-repo owner/nexus-infinite
    
    # List artifacts from private repo
    python scripts/repo_sync.py --list-artifacts \\
        --repo owner/nexus-infinite-core --token $TOKEN
    
    # Download specific artifact
    python scripts/repo_sync.py --download-artifact <name> \\
        --repo owner/nexus-infinite-core --token $TOKEN --output ./models/
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any


def github_api_request(url: str, token: str, method: str = "GET",
                       data: Optional[bytes] = None,
                       headers: Optional[Dict[str, str]] = None) -> Any:
    """Make authenticated GitHub API request"""
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    
    if data:
        req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, data=data, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"ERROR: HTTP {e.code} - {e.reason}")
        print(f"Response: {e.read().decode()[:500]}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def dispatch_workflow(repo: str, token: str, workflow_file: str,
                      ref: str = "main", inputs: Optional[Dict] = None) -> bool:
    """Trigger workflow_dispatch event on target repo"""
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches"
    
    payload = {
        "ref": ref,
        "inputs": inputs or {}
    }
    
    data = json.dumps(payload).encode()
    
    req = urllib.request.Request(url, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, data=data, timeout=30) as resp:
            print(f"Dispatched {workflow_file} on {repo} (ref: {ref})")
            print(f"Inputs: {inputs}")
            return resp.status == 204
    except urllib.error.HTTPError as e:
        print(f"ERROR dispatching workflow: HTTP {e.code}")
        print(f"Response: {e.read().decode()[:500]}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def send_repository_dispatch(repo: str, token: str, event_type: str,
                             client_payload: Optional[Dict] = None) -> bool:
    """Send repository_dispatch event"""
    url = f"https://api.github.com/repos/{repo}/dispatches"
    
    payload = {
        "event_type": event_type,
        "client_payload": client_payload or {}
    }
    
    data = json.dumps(payload).encode()
    
    req = urllib.request.Request(url, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, data=data, timeout=30) as resp:
            print(f"Sent repository_dispatch '{event_type}' to {repo}")
            print(f"Payload: {client_payload}")
            return resp.status == 204
    except urllib.error.HTTPError as e:
        print(f"ERROR sending dispatch: HTTP {e.code}")
        print(f"Response: {e.read().decode()[:500]}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def list_artifacts(repo: str, token: str, run_id: Optional[int] = None) -> list:
    """List artifacts from a repository or specific run"""
    if run_id:
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
    else:
        url = f"https://api.github.com/repos/{repo}/actions/artifacts"
    
    result = github_api_request(url, token)
    if result and "artifacts" in result:
        return result["artifacts"]
    return []


def download_artifact(artifact: Dict, token: str, output_dir: str) -> str:
    """Download and extract a single artifact"""
    url = artifact["archive_download_url"]
    name = artifact["name"]
    
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    
    output_path = Path(output_dir) / f"{name}.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(output_path, "wb") as f:
                f.write(resp.read())
        
        # Extract
        extract_dir = Path(output_dir) / name
        with zipfile.ZipFile(output_path, "r") as z:
            z.extractall(extract_dir)
        
        output_path.unlink()  # Remove zip
        print(f"Downloaded: {name} -> {extract_dir}")
        return str(extract_dir)
        
    except Exception as e:
        print(f"ERROR downloading {name}: {e}")
        return ""


def create_commit(repo: str, token: str, file_path: str, content: bytes,
                  message: str, branch: str = "main") -> bool:
    """Create or update a file in target repo via GitHub API"""
    # First get current file SHA (if exists)
    get_url = f"https://api.github.com/repos/{repo}/contents/{file_path}?ref={branch}"
    
    sha = None
    try:
        req = urllib.request.Request(get_url)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            sha = data.get("sha")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            pass  # File doesn't exist yet
        else:
            print(f"ERROR checking file: HTTP {e.code}")
    
    # Create/update
    import base64
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    
    payload = {
        "message": message,
        "content": base64.b64encode(content).decode(),
        "branch": branch
    }
    if sha:
        payload["sha"] = sha
    
    data = json.dumps(payload).encode()
    
    req = urllib.request.Request(url, method="PUT")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, data=data, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            print(f"Committed {file_path} to {repo}:{branch}")
            return True
    except urllib.error.HTTPError as e:
        print(f"ERROR committing: HTTP {e.code}")
        print(f"Response: {e.read().decode()[:500]}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def sync_public_to_private(args):
    """Trigger training cycle on private repo from public build"""
    print("=" * 60)
    print("SYNC: Public -> Private (Trigger Evolution Cycle)")
    print("=" * 60)
    
    inputs = {
        "public_run_id": str(args.run_id or ""),
        "public_commit": args.commit_sha or "",
        "binary_artifact": args.artifact_name or "nexus-build",
        "trigger_source": "public_scheduler"
    }
    
    if args.dispatch_method == "workflow":
        success = dispatch_workflow(
            args.target_repo, args.token,
            "core_scheduler.yml",
            ref=args.target_branch,
            inputs=inputs
        )
    else:
        success = send_repository_dispatch(
            args.target_repo, args.token,
            "evolution-cycle-triggered",
            client_payload=inputs
        )
    
    return success


def sync_private_to_public(args):
    """Sync best weights from private to public repo"""
    print("=" * 60)
    print("SYNC: Private -> Public (Release Weights)")
    print("=" * 60)
    
    # Read weights file
    weights_path = Path(args.weights_file)
    if not weights_path.exists():
        print(f"ERROR: Weights file not found: {weights_path}")
        return False
    
    with open(weights_path, "rb") as f:
        content = f.read()
    
    # Commit to public repo
    success = create_commit(
        args.target_repo, args.token,
        f"data/nnue/{weights_path.name}",
        content,
        message=f"Sync: {weights_path.name} from private core [auto]",
        branch=args.target_branch
    )
    
    if success and args.create_pr:
        # Create PR for review (optional)
        print("PR creation not implemented in sync script (use GitHub Actions)")
    
    # Also update ELO_DASHBOARD if provided
    if args.elo_history:
        elo_path = Path(args.elo_history)
        if elo_path.exists():
            with open(elo_path, "rb") as f:
                elo_content = f.read()
            create_commit(
                args.target_repo, args.token,
                "ELO_DASHBOARD.md",  # Or elo_history.json
                elo_content,
                message="Update Elo history [auto]",
                branch=args.target_branch
            )
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Nexus Infinite Repo Sync")
    
    # Direction
    parser.add_argument("--direction", choices=[
        "public-to-private", "private-to-public",
        "list-artifacts", "download-artifact"
    ], required=True)
    
    # Auth
    parser.add_argument("--token", required=True, help="GitHub PAT")
    parser.add_argument("--target-repo", required=True, help="owner/repo")
    parser.add_argument("--target-branch", default="main")
    
    # Public -> Private
    parser.add_argument("--run-id", type=int, help="Public Actions run ID")
    parser.add_argument("--commit-sha", help="Git commit SHA")
    parser.add_argument("--artifact-name", default="nexus-build",
                        help="Binary artifact name")
    parser.add_argument("--dispatch-method", choices=["workflow", "repository"],
                        default="repository", help="Dispatch method")
    
    # Private -> Public
    parser.add_argument("--weights-file", help="Path to .nnue file")
    parser.add_argument("--elo-history", help="Path to elo_history.json")
    parser.add_argument("--create-pr", action="store_true",
                        help="Create PR instead of direct commit")
    
    # Artifact operations
    parser.add_argument("--repo", help="Repo for artifact operations")
    parser.add_argument("--artifact-id", help="Artifact ID or name")
    parser.add_argument("--output", default="./downloads", help="Download dir")
    
    args = parser.parse_args()
    
    # Validate token
    if not args.token:
        print("ERROR: --token is required")
        sys.exit(1)
    
    # Route
    if args.direction == "public-to-private":
        success = sync_public_to_private(args)
        sys.exit(0 if success else 1)
    
    elif args.direction == "private-to-public":
        success = sync_private_to_public(args)
        sys.exit(0 if success else 1)
    
    elif args.direction == "list-artifacts":
        repo = args.repo or args.target_repo
        artifacts = list_artifacts(repo, args.token)
        print(f"Found {len(artifacts)} artifacts:")
        for a in artifacts[:20]:  # Limit output
            print(f"  - {a['name']} ({a['size_in_bytes']} bytes, "
                  f"created: {a['created_at']})")
    
    elif args.direction == "download-artifact":
        repo = args.repo or args.target_repo
        artifacts = list_artifacts(repo, args.token)
        
        target = args.artifact_id
        for a in artifacts:
            if a["name"] == target or str(a["id"]) == target:
                download_artifact(a, args.token, args.output)
                break
        else:
            print(f"Artifact not found: {target}")
            sys.exit(1)


if __name__ == "__main__":
    main()
