# Running this bot inside iSH (iPhone)

iSH is Alpine Linux on iOS. It works for light testing but is **not** ideal for long collectors or heavy ML training. iOS will kill background processes.

## 1. Install Python + basic tools in iSH

Open iSH and run:

```sh
apk update
apk add python3 py3-pip python3-dev gcc g++ musl-dev libffi-dev openssl-dev
```

## 2. Get the project files into iSH

**Easiest ways:**

### Option A – Share via Files app
1. Download the zip on your iPhone
2. Open the Files app → On My iPhone
3. In iSH run:
```sh
cd ~
cp -r /mnt/Documents/trading_bot_ish/* ~/bot/
# or wherever you put the unzipped folder
```

### Option B – Download directly inside iSH
If you host the zip somewhere:
```sh
cd ~
wget -O bot.zip "YOUR_DIRECT_LINK"
unzip bot.zip
cd bot
```

### Option C – Use a-Shell or Working Copy to transfer

## 3. Install Python packages

Inside the project folder:

```sh
cd ~/bot          # or wherever you put it
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests pandas numpy joblib python-dotenv
```

**scikit-learn is heavy.** On many iPhones it will fail or be extremely slow.  
Try:
```sh
pip install scikit-learn
```
If it fails or is too slow, the paper trader + scanner still work without the trained model.

**solders** (needed for real swaps):
```sh
pip install solders
```
This often fails on iSH because of Rust compilation. Real `--execute` swaps are better done on a real computer/VPS.

## 4. What actually works well on iSH

| Component              | Works on iSH?          | Notes                                      |
|------------------------|------------------------|--------------------------------------------|
| scanner.py             | Yes                    | Good for quick tests                       |
| collector.py           | Limited                | iOS kills long background runs             |
| paper_trader.py        | Partial                | Works for short sessions                   |
| train.py / brain       | Poor                   | sklearn is heavy + slow                    |
| swap_bot.py (quote)    | Yes                    | Dry-run quotes work                        |
| swap_bot.py (--execute)| Usually No             | solders hard to install + risk             |
| dashboard.py           | Possible               | Use `--host 0.0.0.0` carefully             |

## 5. Recommended workflow on iPhone

1. Use iSH only for **quick scans and quotes**
2. Run the real collector + training on a VPS or Mac/PC
3. Control the VPS from your iPhone with **Termius** or **Blink Shell**

## 6. Quick test commands (inside the project)

```sh
source .venv/bin/activate

# Test scanner
python3 -c "from scanner import scan; print(len(scan()))"

# Quote only (safe)
python3 swap_bot.py --from SOL --to USDC --amount 0.01

# Self-test signing (no network)
python3 swap_bot.py --selftest
```

## 7. Keep it alive longer (still limited)

```sh
# Run in background (still can be killed by iOS)
nohup python3 collector.py > collector.log 2>&1 &
```

Or use `tmux` / `screen` if you install them:
```sh
apk add tmux
tmux
python3 collector.py
# Ctrl+B then D to detach
```

## Warning

- Do **not** put your main wallet private key on the iPhone.
- iSH file system can be wiped if you delete the app.
- For anything serious, use a VPS + SSH from your phone.
