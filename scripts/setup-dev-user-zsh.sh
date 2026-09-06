#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: sudo scripts/setup-dev-user-zsh.sh <username>

Creates a Linux user for this host with:
- default shell: /bin/zsh
- supplementary groups: wheel,docker
- Oh My Zsh installed for that user
- zsh-autosuggestions and zsh-syntax-highlighting plugins installed

What this script intentionally does NOT do:
- clone repositories
- create ~/repos
- run vattention Docker bootstrap scripts
USAGE
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root (use sudo)." >&2
    exit 1
fi

USERNAME="$1"

if ! id "$USERNAME" >/dev/null 2>&1; then
    if [[ ! -x /bin/zsh ]]; then
        echo "Missing /bin/zsh. Install zsh first, then rerun." >&2
        exit 1
    fi

    useradd -m -s /bin/zsh "$USERNAME"
    echo "Created user: $USERNAME"
else
    echo "User already exists: $USERNAME"
    usermod -s /bin/zsh "$USERNAME"
    echo "Ensured default shell is /bin/zsh"
fi

usermod -aG wheel,docker "$USERNAME"
echo "Added $USERNAME to groups: wheel,docker"

# Prompt for password only if account is currently locked.
if passwd -S "$USERNAME" 2>/dev/null | awk '{print $2}' | grep -q '^L$'; then
    echo "Set password for $USERNAME:"
    passwd "$USERNAME"
else
    echo "Password appears set already; skipping password prompt."
fi

# Force password reset at first login.
chage -d 0 "$USERNAME"
echo "Password for $USERNAME is marked expired (must be changed at first login)."

USER_HOME="$(getent passwd "$USERNAME" | cut -d: -f6)"
if [[ -z "$USER_HOME" || ! -d "$USER_HOME" ]]; then
    echo "Could not determine home directory for $USERNAME" >&2
    exit 1
fi

runuser -l "$USERNAME" -c 'set -euo pipefail
if [[ ! -d "$HOME/.oh-my-zsh" ]]; then
    RUNZSH=no CHSH=no KEEP_ZSHRC=yes sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
fi

ZSH_CUSTOM_DIR="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"
mkdir -p "$ZSH_CUSTOM_DIR/plugins"

if [[ ! -d "$ZSH_CUSTOM_DIR/plugins/zsh-autosuggestions" ]]; then
    git clone https://github.com/zsh-users/zsh-autosuggestions "$ZSH_CUSTOM_DIR/plugins/zsh-autosuggestions"
fi

if [[ ! -d "$ZSH_CUSTOM_DIR/plugins/zsh-syntax-highlighting" ]]; then
    git clone https://github.com/zsh-users/zsh-syntax-highlighting "$ZSH_CUSTOM_DIR/plugins/zsh-syntax-highlighting"
fi

ZSHRC="$HOME/.zshrc"
if [[ -f "$ZSHRC" ]]; then
    if grep -q "^plugins=(" "$ZSHRC"; then
        sed -i "s/^plugins=(.*/plugins=(git docker zsh-autosuggestions zsh-syntax-highlighting)/" "$ZSHRC"
    else
        printf "\nplugins=(git docker zsh-autosuggestions zsh-syntax-highlighting)\n" >> "$ZSHRC"
    fi
else
    cat > "$ZSHRC" <<"ZRC"
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"
plugins=(git docker zsh-autosuggestions zsh-syntax-highlighting)
source "$ZSH/oh-my-zsh.sh"
ZRC
fi
'

echo
echo "Setup complete for $USERNAME."
echo "Next steps for that user (run as $USERNAME):"
echo "  mkdir -p ~/repos && cd ~/repos"
echo "  git clone https://github.com/Anodyine/vattention.git"
echo "  cd vattention"
echo "  scripts/docker/build-image.sh"
echo "  scripts/docker/create-container.sh"
echo "  scripts/docker/bootstrap-workspace.sh"
echo
echo "If wheel sudo is not enabled on this host yet, uncomment this line in visudo:"
echo "  %wheel ALL=(ALL:ALL) ALL"
