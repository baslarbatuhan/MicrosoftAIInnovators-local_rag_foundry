"""
Foundry Local asistanı için veri toplama scripti.

Kaynaklar (scraping yok — resmi GitHub içerik servisleri):
  1. MicrosoftDocs/azure-ai-docs → articles/foundry-local/ (MS Learn kaynağı, CC-BY-4.0)
     - Sayfalara gömülen [!INCLUDE] parçaları çözülür (dil-bazlı SDK referansları
       includes/ altında yaşıyor, asıl içerik orada)
     - YAML frontmatter'dan başlık + tarih alınır, zone-pivot işaretleri temizlenir
  2. microsoft/foundry-local → README.md + docs/ (MIT)
  3. Yerel: kurulu foundry CLI'nin --help çıktıları (v0.8.119'dan birebir doğru)

Çıktı: knowledge_bases/foundry/documents/*.txt (ilk satır `Kaynak: <url>` — ingest.py formatı).
İçerik-hash de-duplication: aynı gövde iki kez yazılmaz (includes/reorg mükerrerlerine karşı).
Yeniden çalıştırmak güvenli: klasörü temizleyip yeniden doldurur.
"""
import hashlib
import os
import re
import subprocess
import sys

import requests

DOCS_REPO = "MicrosoftDocs/azure-ai-docs"
DOCS_ROOT = "articles/foundry-local"
PRODUCT_REPO = "microsoft/foundry-local"
OUT_DIR = "knowledge_bases/foundry/documents"

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
INCLUDE_RE = re.compile(r"\[!INCLUDE\s*\[[^\]]*\]\(([^)]+)\)\]")
ZONE_LINE_RE = re.compile(r"^:::.*$", re.MULTILINE)


def github_json(url: str):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def raw_url(repo: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/main/{path}"


def fetch_raw(repo: str, path: str) -> str:
    resp = requests.get(raw_url(repo, path), timeout=30)
    resp.raise_for_status()
    return resp.text


def list_docs_tree() -> list[str]:
    """articles/foundry-local altındaki tüm .md yollarını döner (2 API çağrısı)."""
    contents = github_json(f"https://api.github.com/repos/{DOCS_REPO}/contents/articles")
    sha = next(i["sha"] for i in contents if i["name"] == "foundry-local")
    tree = github_json(f"https://api.github.com/repos/{DOCS_REPO}/git/trees/{sha}?recursive=1")
    return [t["path"] for t in tree["tree"] if t["path"].endswith(".md")]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """YAML frontmatter'ı ayırır; (meta, gövde) döner. Basit anahtar: değer ayrıştırma."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t", "-")):
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, text[match.end():]


def resolve_includes(body: str, page_rel_path: str, files: dict[str, str],
                     used_includes: set[str], depth: int = 0) -> str:
    """[!INCLUDE [ad](yol)] direktiflerini include dosyasının işlenmiş gövdesiyle değiştirir."""
    if depth > 3:  # döngü güvenliği
        return body

    page_dir = os.path.dirname(page_rel_path)

    def replace(match: re.Match) -> str:
        include_path = os.path.normpath(os.path.join(page_dir, match.group(1))).replace("\\", "/")
        if include_path not in files:
            return ""  # foundry-local ağacı dışına işaret ediyor — at
        used_includes.add(include_path)
        _, include_body = parse_frontmatter(files[include_path])
        return resolve_includes(include_body, include_path, files, used_includes, depth + 1)

    return INCLUDE_RE.sub(replace, body)


def clean(body: str) -> str:
    body = ZONE_LINE_RE.sub("", body)                 # ::: zone pivot işaretleri
    body = re.sub(r"\n{3,}", "\n\n", body)            # fazla boş satırları sıkıştır
    return body.strip()


def out_name(rel_path: str) -> str:
    return rel_path.removesuffix(".md").replace("/", "__") + ".txt"


def write_doc(seen_hashes: set[str], filename: str, source_url: str,
              title: str, date: str, body: str) -> bool:
    digest = hashlib.sha256(body.encode()).hexdigest()
    if digest in seen_hashes or len(body) < 200:      # mükerrer veya kırıntı — atla
        return False
    seen_hashes.add(digest)

    header = f"Kaynak: {source_url}\n\n"
    if title:
        header += f"# {title}\n"
    if date:
        header += f"(Doküman tarihi: {date})\n"
    with open(os.path.join(OUT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(header + "\n" + body)
    return True


def collect_ms_learn(seen_hashes: set[str]) -> int:
    print("MS Learn dokümanları indiriliyor...")
    paths = list_docs_tree()
    files = {p: fetch_raw(DOCS_REPO, f"{DOCS_ROOT}/{p}") for p in paths}
    print(f"  {len(files)} .md dosyası indirildi")

    used_includes: set[str] = set()
    written = 0

    # 1. Sayfa dosyaları (includes/ dışındakiler) — include'ları gömerek yaz
    for path, text in files.items():
        if path.startswith("includes/"):
            continue
        meta, body = parse_frontmatter(text)
        body = clean(resolve_includes(body, path, files, used_includes))
        url = f"https://github.com/{DOCS_REPO}/blob/main/{DOCS_ROOT}/{path}"
        if write_doc(seen_hashes, out_name(path), url, meta.get("title", ""), meta.get("ms.date", ""), body):
            written += 1

    # 2. Hiçbir sayfaya gömülmemiş (yetim) include'lar — kaybolmasınlar diye tek başına yaz
    for path, text in files.items():
        if not path.startswith("includes/") or path in used_includes:
            continue
        meta, body = parse_frontmatter(text)
        body = clean(resolve_includes(body, path, files, used_includes))
        url = f"https://github.com/{DOCS_REPO}/blob/main/{DOCS_ROOT}/{path}"
        if write_doc(seen_hashes, out_name(path), url, meta.get("title", ""), meta.get("ms.date", ""), body):
            written += 1

    return written


def collect_product_repo(seen_hashes: set[str]) -> int:
    print("Ürün reposu (microsoft/foundry-local) indiriliyor...")
    written = 0
    for path in ["README.md", "docs/Structured Outputs.md"]:
        try:
            text = fetch_raw(PRODUCT_REPO, path)
        except requests.HTTPError as exc:
            print(f"  [uyari] {path} indirilemedi: {exc}")
            continue
        _, body = parse_frontmatter(text)
        url = f"https://github.com/{PRODUCT_REPO}/blob/main/{path}"
        name = "product-repo__" + out_name(path.replace(" ", "-"))
        if write_doc(seen_hashes, name, url, "", "", clean(body)):
            written += 1
    return written


def collect_cli_help(seen_hashes: set[str]) -> int:
    """Kurulu foundry CLI'nin yardım metinlerini tek dokümanda topla (sıfır indirme)."""
    print("CLI yardım metinleri toplanıyor...")
    sections = []
    for args in [["--help"], ["model", "--help"], ["service", "--help"], ["cache", "--help"]]:
        try:
            result = subprocess.run(["foundry", *args], capture_output=True, text=True, timeout=60)
            output = (result.stdout or "") + (result.stderr or "")
            if output.strip():
                sections.append(f"## foundry {' '.join(args)}\n\n{output.strip()}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"  [uyari] foundry {' '.join(args)} çalıştırılamadı: {exc}")

    if not sections:
        return 0
    body = "Foundry Local CLI help output (from installed CLI).\n\n" + "\n\n".join(sections)
    return 1 if write_doc(seen_hashes, "cli-help-output.txt", "foundry CLI v0.8.119 (yerel çıktı)",
                          "Foundry Local CLI command reference", "", body) else 0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for old in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, old))

    seen_hashes: set[str] = set()
    total = 0
    total += collect_ms_learn(seen_hashes)
    total += collect_product_repo(seen_hashes)
    total += collect_cli_help(seen_hashes)

    sizes = [os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR)]
    print(f"\nTAMAM: {total} doküman → {OUT_DIR}/ (toplam {sum(sizes)} bayt)")


if __name__ == "__main__":
    main()
