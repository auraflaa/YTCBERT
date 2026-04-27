import yt_dlp
import logging

url = "https://www.youtube.com/watch?v=7XuaRjB4OBk"

clients = ['web', 'web,tv', 'tv', 'ios', 'android', 'mweb']

for client in clients:
    print(f"\n--- Testing Client: {client} ---")
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'quiet': True,
        'impersonate': 'chrome',
        'extractor_args': {'youtube': [f'client={client}']}
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✅ SUCCESS: {client}")
    except Exception as e:
        print(f"❌ FAILED: {client} | Error: {e}")
