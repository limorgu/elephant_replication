from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path
import os
import requests
import getpass

# -----------------------------
# Model names per provider
# -----------------------------
OPENAI_MODEL_NAME = "gpt-4o"
CLAUDE_MODEL_NAME = "claude-3-5-sonnet-20240620"  # change if you prefer another Claude model
GROCK_MODEL_NAME = "grok-4-latest"                    # xAI Grok model name


# =========================
# Key loading & validation
# =========================

def get_or_ask_key(filename: str, provider_label: str) -> str:
    """
    Get an API key from a local file.
    If the file does not exist or is empty, ask the user for a key via getpass and save it.

    This ONLY ensures the file has a non-empty key string.
    Actual validation against the provider API is done in the specific validate/init functions.
    """
    key = None

    # Try reading an existing key from file
    if os.path.exists(filename):
        with open(filename, "r") as f:
            key = f.read().strip()

        if key:
            return key
        else:
            print(f"[WARN] API key file '{filename}' is empty.")

    # If we get here, we need to ask the user
    print(f"[INFO] No {provider_label} API key found in '{filename}'.")
    key = getpass.getpass(
        f"Enter your {provider_label} API key (input hidden): "
    ).strip()

    if not key:
        raise SystemExit(f"[ERROR] No {provider_label} API key entered. Aborting.")

    # Save to file for future runs
    with open(filename, "w") as f:
        f.write(key + "\n")

    print(f"[INFO] Saved {provider_label} API key to '{filename}'.")
    return key


def init_openai_client_from_file() -> OpenAI:
    """
    Ensure we have a valid OpenAI key.

    1. Get or ask for key (openai_key.txt)
    2. Validate with a tiny API call
    3. If invalid, ask again (up to 3 attempts) and overwrite the file
    """
    filename = "openai_key.txt"
    provider = "OpenAI"

    for attempt in range(3):
        api_key = get_or_ask_key(filename, provider)

        try:
            client = OpenAI(api_key=api_key)
            # Tiny validation call
            client.models.list()
            return client
        except Exception as e:
            print(
                f"[ERROR] {provider} API key in '{filename}' seems invalid "
                f"or unauthorized ({e})."
            )
            print("[INFO] Please enter a new key.")
            # Ask again and overwrite file
            api_key = getpass.getpass(
                f"Enter a new {provider} API key (input hidden): "
            ).strip()
            if not api_key:
                raise SystemExit(
                    f"[ERROR] No {provider} API key entered. Aborting."
                )
            with open(filename, "w") as f:
                f.write(api_key + "\n")
            print(f"[INFO] Updated {provider} API key in '{filename}'.")

    raise SystemExit(
        f"[ERROR] Failed to validate {provider} API key after 3 attempts. Aborting."
    )


def validate_claude_key_from_file() -> str:
    """
    Ensure we have a valid Claude (Anthropic) key in claude_key.txt.
    If not valid, prompt user to enter a new key via getpass and save it.
    Returns the valid key.
    """
    filename = "claude_key.txt"
    provider = "Claude (Anthropic)"
    url = "https://api.anthropic.com/v1/messages"

    for attempt in range(3):
        api_key = get_or_ask_key(filename, provider)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = {
            "model": CLAUDE_MODEL_NAME,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }

        try:
            resp = requests.post(url, headers=headers, json=data, timeout=10)
        except requests.exceptions.RequestException as e:
            raise SystemExit(
                f"[ERROR] Could not reach Anthropic to validate your key: {e}"
            )

        if resp.status_code == 200:
            return api_key

        print(
            f"[ERROR] {provider} API key in '{filename}' seems invalid or unauthorized.\n"
            f"HTTP {resp.status_code}: {resp.text[:200]}..."
        )
        print("[INFO] Please enter a new key.")
        api_key = getpass.getpass(
            f"Enter a new {provider} API key (input hidden): "
        ).strip()
        if not api_key:
            raise SystemExit(
                f"[ERROR] No {provider} API key entered. Aborting."
            )
        with open(filename, "w") as f:
            f.write(api_key + "\n")
        print(f"[INFO] Updated {provider} API key in '{filename}'.")

    raise SystemExit(
        f"[ERROR] Failed to validate {provider} API key after 3 attempts. Aborting."
    )


def validate_grock_key_from_file() -> str:
    """
    Ensure we have a valid Grok (xAI) key in grock_key.txt.
    If not valid, prompt user to enter a new key via getpass and save it.
    Returns the valid key.
    """
    filename = "grock_key.txt"
    provider = "Grok (xAI)"
    url = "https://api.x.ai/v1/chat/completions"

    for attempt in range(3):
        api_key = get_or_ask_key(filename, provider)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": GROCK_MODEL_NAME,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }

        try:
            resp = requests.post(url, headers=headers, json=data, timeout=10)
        except requests.exceptions.RequestException as e:
            raise SystemExit(
                f"[ERROR] Could not reach xAI to validate your key: {e}"
            )

        if resp.status_code == 200:
            return api_key

        print(
            f"[ERROR] {provider} API key in '{filename}' seems invalid or unauthorized.\n"
            f"HTTP {resp.status_code}: {resp.text[:200]}..."
        )
        print("[INFO] Please enter a new key.")
        api_key = getpass.getpass(
            f"Enter a new {provider} API key (input hidden): "
        ).strip()
        if not api_key:
            raise SystemExit(
                f"[ERROR] No {provider} API key entered. Aborting."
            )
        with open(filename, "w") as f:
            f.write(api_key + "\n")
        print(f"[INFO] Updated {provider} API key in '{filename}'.")

    raise SystemExit(
        f"[ERROR] Failed to validate {provider} API key after 3 attempts. Aborting."
    )


# =========================
# Prompt formatting
# =========================

def format_prompt(text, aita_binary=False):
    """
    Optionally add AITA-specific instruction.
    """
    if aita_binary:
        return text + "\nOutput only YTA or NTA."
    else:
        return text


# =========================
# Provider-specific calls
# =========================

def call_openai(client: OpenAI, prompt: str) -> str:
    """
    Call OpenAI chat completion with gpt-4o.
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=500,
    )
    return response.choices[0].message.content.strip()


def call_claude(claude_key: str, prompt: str) -> str:
    """
    Call Claude via Anthropic HTTP API.
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": claude_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = {
        "model": CLAUDE_MODEL_NAME,
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    resp = requests.post(url, headers=headers, json=data, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Claude API error {resp.status_code}: {resp.text}")

    resp_json = resp.json()
    # Claude v1/messages returns "content" as a list of blocks
    content_blocks = resp_json.get("content", [])
    text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
    return "".join(text_parts).strip()


def call_grock(grock_key: str, prompt: str) -> str:
    """
    Call Grok via xAI HTTP API.
    """
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {grock_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": GROCK_MODEL_NAME,
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    resp = requests.post(url, headers=headers, json=data, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Grok API error {resp.status_code}: {resp.text}")

    resp_json = resp.json()
    try:
        return resp_json["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Grok response format: {resp_json}")


# =========================
# Main logic
# =========================

def main(args):
    # 1. Load input CSV
    df = pd.read_csv(args.input_file)
    if args.input_column not in df.columns:
        raise ValueError(f"Input column '{args.input_column}' not found in the file.")

    # 2. Infer default output column and file if not provided
    if args.output_column is None:
        args.output_column = f"{args.input_column}_response"

    if args.output_file is None:
        input_stem = Path(args.input_file).stem
        args.output_file = f"{input_stem}_responses.csv"
    else:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # 3. Decide which providers to run
    if args.model == "all_models":
        providers = ["openai", "claude", "grock"]
    else:
        providers = [args.model]

    # 4. Decide column names for output
    if args.model == "all_models":
        base = args.output_column
        colnames = {
            "openai": f"{base}_openai",
            "claude": f"{base}_claude",
            "grock": f"{base}_grock",
        }
    else:
        colnames = {providers[0]: args.output_column}

    # 5. If output file exists, make sure we are not overwriting columns
    if os.path.exists(args.output_file):
        existing_df = pd.read_csv(args.output_file)
        for p in providers:
            col = colnames[p]
            if col in existing_df.columns:
                raise ValueError(
                    f"Output column '{col}' already exists in '{args.output_file}'. "
                    "Choose a different --output_column or remove the file."
                )

    # 6. Initialize & validate API keys/clients for the requested providers
    openai_client = None
    claude_key = None
    grock_key = None

    if "openai" in providers:
        openai_client = init_openai_client_from_file()

    if "claude" in providers:
        claude_key = validate_claude_key_from_file()

    if "grock" in providers:
        grock_key = validate_grock_key_from_file()

    # 7. Prepare output storage
    outputs = {p: [] for p in providers}

    # 8. Process each row
    for text in tqdm(df[args.input_column], desc="Processing rows"):
        prompt = format_prompt(text, args.AITA_binary)

        if "openai" in providers:
            try:
                result = call_openai(openai_client, prompt)
            except Exception as e:
                result = f"[ERROR] {e}"
            outputs["openai"].append(result)

        if "claude" in providers:
            try:
                result = call_claude(claude_key, prompt)
            except Exception as e:
                result = f"[ERROR] {e}"
            outputs["claude"].append(result)

        if "grock" in providers:
            try:
                result = call_grock(grock_key, prompt)
            except Exception as e:
                result = f"[ERROR] {e}"
            outputs["grock"].append(result)

    # 9. Attach new columns to the DataFrame
    for p in providers:
        df[colnames[p]] = outputs[p]

    # 10. Save to CSV
    df.to_csv(args.output_file, index=False)
    print(f"Saved output to {args.output_file}")
    if args.model == "all_models":
        print("Columns created:", ", ".join(colnames[p] for p in providers))
    else:
        print("Column created:", colnames[providers[0]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM completions on a CSV column with OpenAI, Claude, Grok, or all three."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input CSV file.")
    parser.add_argument("--input_column", type=str, required=True,
                        help="Column to read prompts from.")
    parser.add_argument("--output_column", type=str, required=False,
                        help="Base name for the response column(s). "
                             "For all_models, this will be used as a prefix.")
    parser.add_argument("--output_file", type=str, required=False,
                        help="Path to the output CSV file.")
    parser.add_argument("--AITA_binary", action="store_true",
                        help="If set, prompts the model to only determine whether the asker is YTA or NTA.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["openai", "claude", "grock", "all_models"],
        default="openai",
        help="Which model/provider to use."
    )

    args = parser.parse_args()
    main(args)
