#!/usr/bin/env python3
"""Validate all servers in servers.json are reachable and respond."""
import json, urllib.request, urllib.error, sys, time

def test_server(url, model, timeout=60):
    """Ask the model 'Are you there?' — if it responds, the model is loaded and working."""
    try:
        req = urllib.request.Request(f"{url}/api/generate",
            data=json.dumps({"model": model, "prompt": "Are you there?", "stream": False}).encode(),
            headers={"Content-Type": "application/json"})
        start = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            elapsed = time.time() - start
            return ("OK", f"{elapsed:.1f}s", data.get("response", "")[:60])
    except urllib.error.HTTPError as e:
        return ("HTTP_ERR", str(e.code), str(e.reason))
    except urllib.error.URLError as e:
        return ("UNREACHABLE", str(e.reason), "")
    except json.JSONDecodeError:
        return ("BAD_JSON", "", "")
    except Exception as e:
        return ("ERROR", str(e), "")

def main():
    with open("servers.json") as f:
        servers = json.load(f)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"{'Name':<20} {'Status':<14} {'Time':<8} {'Response'}")
    print("-" * 80)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {}
        for s in servers:
            name = s.get("name", "?")
            if not s.get("enabled", False):
                print(f"{name:<20} {'SKIPPED':<14} {'':<8} (disabled)")
                continue
            url = s.get("url", "")
            model = s.get("model", "")
            futures[pool.submit(test_server, url, model)] = name
        for f in as_completed(futures):
            name = futures[f]
            status, detail, response = f.result()
            if status == "OK":
                ok += 1
            else:
                fail += 1
            print(f"{name:<20} {status:<14} {detail:<8} {response[:40]}")

    enabled = sum(1 for s in servers if s.get("enabled", False))
    disabled = len(servers) - enabled
    print(f"\n{ok}/{enabled} enabled servers OK, {fail} unreachable, {disabled} disabled")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
