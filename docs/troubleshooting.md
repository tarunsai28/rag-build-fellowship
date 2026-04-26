# Troubleshooting

Symptoms students hit most often, with the actual fix.

## `GEMINI_API_KEY is not set`

You haven't created `.env`, or you created it but didn't paste the key.

```bash
cp .env.example .env
# open .env, paste GEMINI_API_KEY=...
uv run python scripts/verify_setup.py
```

## `ResourceExhausted: 429 ... quota exceeded`

You've hit Gemini's free-tier rate limit. The retry decorator handles
short bursts; for sustained traffic, slow down or switch providers:

```bash
# Option A: switch to Ollama (fully local, see SETUP.md)
echo "LLM_PROVIDER=ollama" >> .env
echo "EMBEDDINGS_PROVIDER=ollama" >> .env

# Option B: switch to OpenAI (paid, but no shared quota)
echo "LLM_PROVIDER=openai" >> .env
```

## `chromadb` import works, but `count()` returns 0 after ingest

The collection name in `.env` doesn't match what was indexed. Either set
`CHROMA_COLLECTION` to the original name, or wipe and re-index:

```bash
rm -rf .chroma
uv run python scripts/ingest.py
```

## `pypdf` crashes on a specific PDF

Some PDFs ship without an extractable text layer (image-only scans).
`DocumentLoader._load_pdf` will catch the error and skip the file. To
confirm, run:

```bash
uv run python -c "from pypdf import PdfReader; print(PdfReader('data/sample_docs/your.pdf').pages[0].extract_text())"
```

If output is empty, you'll need OCR (out of scope for this project).

## API server starts but `/ask` returns 500

Check the server logs — usually one of:
- Provider key missing (same fix as above).
- `.chroma/` is empty. Run `POST /ingest` or `scripts/ingest.py`.

## On Windows: `'uv' is not recognized`

You installed `uv` but haven't restarted the shell. Close the PowerShell
window and reopen, or `refreshenv` if you have Chocolatey.

## On Windows: long-path errors during `uv sync`

Enable long path support once, as Administrator:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                 -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## On macOS Apple Silicon: `sentence-transformers` install hangs

Torch downloads can be large. Either be patient, or stick with the default
Gemini embeddings (`EMBEDDINGS_PROVIDER=gemini`) and skip the local extra
entirely.
