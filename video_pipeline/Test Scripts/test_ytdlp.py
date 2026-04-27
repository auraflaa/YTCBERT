import yt_dlp
import requests

def test_proxy_cdn():
    print("\n--- Testing CDN Proxy Bypass ---")
    url = "https://www.youtube.com/watch?v=s3KnSb9b4Pk"
    ydl_opts = {
        'skip_download': True, 'writesubtitles': True, 'writeautomaticsub': True,
        'subtitleslangs': ['en'], 'quiet': True, 'no_warnings': True,
        'extractor_args': {'youtube': ['client=android,ios']}
    }
    
    # 1. Get Signed URL natively (this works because it's the API layer)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        sub_url = info.get('requested_subtitles', {}).get('en', {}).get('url')
        if not sub_url:
            return print("No sub url found!")
            
    if 'fmt=' not in sub_url: sub_url += '&fmt=json3'
    sub_url = sub_url.replace('fmt=vtt', 'fmt=json3')
    
    print("Natively fetched Signed CDN URL.")
    
    # 2. Scrape a quick proxy
    print("Fetching a quick free proxy...")
    try:
        from bs4 import BeautifulSoup
        resp = requests.get("https://free-proxy-list.net/", timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        proxies = []
        for row in soup.find("tbody").find_all("tr")[:10]:
            cols = row.find_all("td")
            if "yes" in cols[6].text.lower():
                proxies.append(f"http://{cols[0].text}:{cols[1].text}")
    except Exception as e:
        return print(f"Failed to get proxy: {e}")
        
    print(f"Found {len(proxies)} proxies. Attempting download...")
    
    # 3. Test Proxy Download
    for p in proxies:
        print(f"Trying {p}...")
        try:
            r = requests.get(sub_url, proxies={"http": p, "https": p}, timeout=5)
            if r.status_code == 200:
                print(f"✅ SUCCESS! Downloaded data length: {len(r.text)}")
                return
            elif r.status_code == 429:
                print(f"❌ 429 - Proxy IP {p} is ALSO banned!")
            else:
                print(f"❌ Error {r.status_code}")
        except Exception:
            pass
            
    print("All proxies failed or timed out.")

if __name__ == "__main__":
    test_proxy_cdn()
