#!/usr/bin/env python3
"""
Parse a named profile from .vscode/sftp.json and output shell-eval-able
connection parameters for use in rsync / scp commands.

Usage:
    python common_utils/parse_sftp_profile.py "PROFILE_NAME"
    python common_utils/parse_sftp_profile.py --list

Output (eval mode, default):
    Prints shell variable assignments that can be eval'd:
        SFTP_HOST="<host>"
        SFTP_PORT="<port>"
        SFTP_USER="<user>"
        SFTP_REMOTE_PATH="<remote_path>"
        SFTP_AUTH_TYPE="key"          # "key" | "password"
        SFTP_KEY_PATH="<key_path>"
        SFTP_PASSWORD=""              # empty for key-based
"""

import argparse
import json
import sys
from pathlib import Path


def find_sftp_json() -> Path:
    """Search for sftp.json in .vscode/ relative to repo root."""
    # Walk up from this script's location to find .vscode/sftp.json
    here = Path(__file__).resolve().parent
    for directory in [here, here.parent, *here.parents]:
        candidate = directory / ".vscode" / "sftp.json"
        if candidate.exists():
            return candidate
        if (directory / ".git").exists():
            break
    raise FileNotFoundError(
        "Could not find .vscode/sftp.json. Run this script from the repo root."
    )


def load_profiles(sftp_json_path: Path) -> dict:
    with sftp_json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    profiles = data.get("profiles", {})
    if not profiles:
        raise ValueError("No 'profiles' key found in sftp.json")
    return profiles


def get_profile(profiles: dict, name: str) -> dict:
    if name not in profiles:
        available = "\n  ".join(profiles.keys())
        raise KeyError(
            f"Profile '{name}' not found in sftp.json.\n"
            f"Available profiles:\n  {available}"
        )
    return profiles[name]


def print_shell_vars(profile: dict, name: str) -> None:
    """Print shell variable assignments for the profile."""
    host = profile.get("host", "")
    port = str(profile.get("port", 22))
    username = profile.get("username", "")
    remote_path = profile.get("remotePath", "")
    password = profile.get("password", "")
    key_path = profile.get("privateKeyPath", "")

    if key_path:
        auth_type = "key"
    elif password:
        auth_type = "password"
    else:
        auth_type = "none"

    lines = [
        f'SFTP_PROFILE={json.dumps(name)}',
        f'SFTP_HOST={json.dumps(host)}',
        f'SFTP_PORT={json.dumps(port)}',
        f'SFTP_USER={json.dumps(username)}',
        f'SFTP_REMOTE_PATH={json.dumps(remote_path)}',
        f'SFTP_AUTH_TYPE={json.dumps(auth_type)}',
        f'SFTP_KEY_PATH={json.dumps(key_path)}',
        f'SFTP_PASSWORD={json.dumps(password)}',
    ]
    print("\n".join(lines))


# rsync filter rules: only sync files needed for result analysis,
# excluding large model weights and tokenizer files.
RSYNC_RESULT_FILTERS = [
    '--include="*/"',
    '--include="*SUMMARY.json"',
    '--include="*EVAL.json"',
    '--include="*.log"',
    '--include=".hydra/***"',
    '--include="trainer_state.json"',
    '--include="adapter_config.json"',
    '--include="README.md"',
    '--exclude="*"',
]


def print_rsync_command(profile: dict, remote_subpath: str = "saves/") -> None:
    """
    Print a ready-to-run rsync command to pull remote_subpath to local.

    remote_subpath: path relative to remotePath on the server (default: "saves/")
    """
    host = profile.get("host", "")
    port = str(profile.get("port", 22))
    username = profile.get("username", "")
    remote_path = profile.get("remotePath", "").rstrip("/")
    password = profile.get("password", "")
    key_path = profile.get("privateKeyPath", "")

    ssh_opts = f"-p {port}"
    if key_path:
        ssh_opts += f" -i {key_path}"

    filter_opts = " \\\n  ".join(RSYNC_RESULT_FILTERS)
    remote_src = f"{username}@{host}:{remote_path}/{remote_subpath}"
    rsync_base = (
        f'rsync -avz --progress \\\n'
        f'  {filter_opts} \\\n'
        f'  -e "ssh {ssh_opts}" {remote_src} ./{remote_subpath}'
    )

    if key_path:
        print(rsync_base)
    else:
        # Use SSHPASS env var to avoid exposing password in process list / shell history
        print(f'SSHPASS={json.dumps(password)} sshpass -e {rsync_base}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse .vscode/sftp.json and output connection info for a named profile."
    )
    parser.add_argument(
        "profile_name",
        nargs="?",
        help="Name of the profile to look up (use --list to see available profiles)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available profile names and exit",
    )
    parser.add_argument(
        "--rsync",
        metavar="SUBPATH",
        nargs="?",
        const="saves/",
        help="Print rsync command instead of shell vars. Optionally specify remote subpath (default: saves/)",
    )
    args = parser.parse_args()

    try:
        sftp_json = find_sftp_json()
        profiles = load_profiles(sftp_json)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        print("Available profiles in .vscode/sftp.json:")
        for name in profiles:
            p = profiles[name]
            auth = "key" if p.get("privateKeyPath") else "password" if p.get("password") else "none"
            print(f"  [{auth:8s}]  {name}")
        return

    if not args.profile_name:
        parser.print_help()
        sys.exit(1)

    try:
        profile = get_profile(profiles, args.profile_name)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.rsync is not None:
        print_rsync_command(profile, args.rsync)
    else:
        print_shell_vars(profile, args.profile_name)


if __name__ == "__main__":
    main()
