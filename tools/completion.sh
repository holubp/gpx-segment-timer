#!/usr/bin/env bash
# Bash completion for gpx-segment-timer.py and tools/debug_match_metrics.py.
# Source this file: . ./tools/completion.sh

_gpx_timer_get_opts() {
  local script="$1"
  if [ ! -f "$script" ]; then
    return 0
  fi
  python3 "$script" --help 2>/dev/null \
    | grep -oE '(-{1,2}[A-Za-z0-9][A-Za-z0-9-]*)' \
    | sort -u \
    | tr '\n' ' '
}

_gpx_timer_complete() {
  local cur script opts
  cur="${COMP_WORDS[COMP_CWORD]}"
  script="${COMP_WORDS[0]}"
  opts="$(_gpx_timer_get_opts "$script")"
  COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
}

_gpx_timer_debug_complete() {
  local cur script opts
  cur="${COMP_WORDS[COMP_CWORD]}"
  script="${COMP_WORDS[0]}"
  opts="$(_gpx_timer_get_opts "$script")"
  COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
}

_gpx_timer_python_complete() {
  local cur script opts
  cur="${COMP_WORDS[COMP_CWORD]}"
  if [ "$COMP_CWORD" -lt 2 ]; then
    return 0
  fi
  script="${COMP_WORDS[1]}"
  case "$script" in
    gpx-segment-timer.py|./gpx-segment-timer.py|*/gpx-segment-timer.py)
      opts="$(_gpx_timer_get_opts "$script")"
      COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
      ;;
    debug_match_metrics.py|./debug_match_metrics.py|*/debug_match_metrics.py)
      opts="$(_gpx_timer_get_opts "$script")"
      COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
      ;;
  esac
}

complete -F _gpx_timer_complete gpx-segment-timer.py ./gpx-segment-timer.py
complete -F _gpx_timer_debug_complete debug_match_metrics.py ./tools/debug_match_metrics.py tools/debug_match_metrics.py
# Opt-in python3 completion (only activates when script path matches).
if [ -n "${GPX_TIMER_PYTHON_COMPLETE:-}" ]; then
  complete -F _gpx_timer_python_complete -o default -o bashdefault python3
fi
