#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
skills_dir="$project_dir/.opencode/skills"

for skill_name in rwe-study rwe-explore; do
  if [[ ! -f "$project_dir/skills/$skill_name/SKILL.md" ]]; then
    echo "Missing bundled skill: skills/$skill_name/SKILL.md" >&2
    exit 1
  fi
done

mkdir -p "$skills_dir"

link_skill() {
  local skill_name="$1"
  local target="../../skills/$skill_name"
  local link="$skills_dir/$skill_name"

  if [[ -L "$link" ]]; then
    if [[ "$(readlink "$link")" == "$target" ]]; then
      echo "OpenCode skill link already present: $skill_name"
      return
    fi
    echo "Refusing to replace existing symlink: $link" >&2
    exit 1
  fi

  if [[ -e "$link" ]]; then
    echo "Refusing to replace existing path: $link" >&2
    exit 1
  fi

  ln -s "$target" "$link"
  echo "Linked OpenCode skill: $skill_name"
}

link_skill rwe-study
link_skill rwe-explore

echo
echo "Next, ensure the bare pheno-rwe command is on PATH and authenticate OpenCode:"
echo "  uv sync --all-extras --dev"
echo "  source .venv/bin/activate"
echo "  opencode auth login"
echo "See docs/testing-with-opencode.md for the synthetic smoke-test workflow."
