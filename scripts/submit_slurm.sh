#!/usr/bin/env bash
# Submit any command as a single-node Slurm batch job run from the repo root.
#
#   scripts/submit_slurm.sh --dry-run -- nvidia-smi
#   scripts/submit_slurm.sh --account <alloc> --job-name mamba-ink -- \
#       uv run --no-sync python <training-script> [args...]
#
# The job runs whatever this checkout contains when it *starts*, not when it
# was submitted, so the log header records the commit that actually ran.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/submit_slurm.sh [options] -- <command> [args...]

Options:
  --partition <name>         Slurm partition (default: rtx-small)
  --time <duration>          Slurm wall time (default: 24:00:00)
  --job-name <name>          Slurm job name (default: ink-tokenizer)
  --account <project>        Optional Slurm account/allocation
  --output <path-pattern>    Combined output path (% placeholders in basename;
                             default: logs/slurm/%x-%j.out)
  --module <name>            Module to load first; repeatable (default: none)
  --sbatch-option=<option>   Additional single sbatch argument; repeatable
  --dry-run                  Print the submission without submitting
  -h, --help                 Show usage

Multi-token Slurm settings need one --sbatch-option=... per token. Prefer
--option=value forms. Options managed by this helper, including GPU/GRES
allocation options, cannot be overridden; pick a GPU partition instead.

Run `uv sync` on the login node first and pass `uv run --no-sync ...` as the
command, so jobs neither hit the network nor race each other over .venv.
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    printf 'Run %q --help for usage.\n' "$0" >&2
    exit 2
}

require_value() {
    local option="$1"
    local remaining="$2"
    local value="${3-}"

    ((remaining >= 2)) && [[ "${value}" != -- ]] || die "${option} requires a value"
    [[ -n "${value}" ]] || die "${option} requires a non-empty value"
}

reject_managed_sbatch_option() {
    local option="$1"
    local option_name
    local managed_option

    [[ "${option}" != : ]] || die \
        "--sbatch-option cannot be the heterogeneous-job separator ':'"

    option_name="${option%%=*}"
    if [[ "${option_name}" = --* ]]; then
        # sbatch accepts unambiguous abbreviations (--part=...), so reject any
        # prefix of a managed option, not just exact names.
        for managed_option in \
            --partition \
            --time \
            --nodes \
            --ntasks \
            --ntasks-per-core \
            --ntasks-per-gpu \
            --ntasks-per-node \
            --ntasks-per-socket \
            --job-name \
            --account \
            --output \
            --error \
            --chdir \
            --workdir \
            --export \
            --gres \
            --gres-flags \
            --gpus \
            --gpus-per-node \
            --gpus-per-socket \
            --gpus-per-task \
            --tres-per-task \
            --wrap \
            --external; do
            if [[ "${managed_option}" = "${option_name}"* ]]; then
                die "--sbatch-option conflicts with helper-managed setting: ${option}"
            fi
        done
    fi

    case "${option}" in
        -p|-p?*|-t|-t?*|-N|-N?*|-n|-n?*|-J|-J?*|\
        -A|-A?*|-o|-o?*|-e|-e?*|-D|-D?*|-G|-G?*)
            die "--sbatch-option conflicts with helper-managed setting: ${option}"
            ;;
    esac
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"

partition="rtx-small"
wall_time="24:00:00"
job_name="ink-tokenizer"
account=""
output=""
modules=()
dry_run=false
delimiter_seen=false
extra_sbatch_options=()

while (($# > 0)); do
    case "$1" in
        --)
            delimiter_seen=true
            shift
            break
            ;;
        --partition|--time|--job-name|--account|--output|--module)
            option="$1"
            require_value "${option}" "$#" "${2-}"
            value="$2"
            shift 2
            case "${option}" in
                --partition) partition="${value}" ;;
                --time) wall_time="${value}" ;;
                --job-name) job_name="${value}" ;;
                --account) account="${value}" ;;
                --output) output="${value}" ;;
                --module) modules+=("${value}") ;;
            esac
            ;;
        --sbatch-option=*)
            sbatch_option="${1#--sbatch-option=}"
            [[ -n "${sbatch_option}" ]] || die "--sbatch-option requires a non-empty value"
            reject_managed_sbatch_option "${sbatch_option}"
            extra_sbatch_options+=("${sbatch_option}")
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option before command delimiter: $1"
            ;;
    esac
done

${delimiter_seen} || die "the -- command delimiter is required"
(($# > 0)) || die "a command is required after --"
command_args=("$@")

if [[ -z "${output}" ]]; then
    output_path="${repo_root}/logs/slurm/%x-%j.out"
elif [[ "${output}" = /* ]]; then
    output_path="${output}"
else
    output_path="${repo_root}/${output}"
fi
output_dir="$(dirname -- "${output_path}")"
if [[ "${output_dir}" = *%* ]]; then
    die "--output supports Slurm % placeholders only in the filename, not its directory"
fi

sbatch_args=(
    "--partition=${partition}"
    "--time=${wall_time}"
    "--nodes=1"
    "--ntasks=1"
    "--job-name=${job_name}"
)
if [[ -n "${account}" ]]; then
    sbatch_args+=("--account=${account}")
fi
sbatch_args+=(
    "--output=${output_path}"
    "--error=${output_path}"
    "--chdir=${repo_root}"
    "--export=ALL"
)
# The guarded expansion keeps an empty array safe under `set -u` on bash < 4.4.
sbatch_args+=(${extra_sbatch_options[@]+"${extra_sbatch_options[@]}"})

payload_lines=(
    '#!/usr/bin/env bash'
    'set -eo pipefail'
)
# Module init scripts read unset variables, so nounset waits until after them.
if ((${#modules[@]} > 0)); then
    printf -v quoted_modules ' %q' "${modules[@]}"
    payload_lines+=("module load${quoted_modules}")
fi
payload_lines+=('set -u')
printf -v quoted_repo_root '%q' "${repo_root}"
printf -v quoted_command ' %q' "${command_args[@]}"
printf -v quoted_banner '%q' "==>${quoted_command}"
# Single-quoted lines expand inside the job, not at submission.
# shellcheck disable=SC2016
payload_lines+=(
    "export INK_TOKENIZER_REPO_ROOT=${quoted_repo_root}"
    'cd -- "$INK_TOKENIZER_REPO_ROOT"'
    'printf "==> job %s on %s at %s\n" "${SLURM_JOB_ID:-?}" "$(hostname)" "$(date -Is)"'
    'printf "==> commit %s\n" "$(git describe --always --dirty 2>/dev/null || echo unknown)"'
    'if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L || true; else echo "==> nvidia-smi not found: this job may be CPU-only"; fi'
    "printf '%s\\n' ${quoted_banner}"
    "exec${quoted_command}"
)
printf -v payload '%s\n' "${payload_lines[@]}"

if ${dry_run}; then
    printf 'Sbatch command (payload is supplied on standard input):\n  '
    printf '%q ' sbatch "${sbatch_args[@]}"
    printf '\nCommand after helper --:\n  '
    printf '%q ' "${command_args[@]}"
    printf '\nBatch payload:\n%s' "${payload}"
    exit 0
fi

mkdir -p -- "${output_dir}"
sbatch "${sbatch_args[@]}" <<<"${payload}"
printf 'Slurm log: %s\n' "${output_path}"
