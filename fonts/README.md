# Figure typography

TPGT bar plots use **Times New Roman** when available (`genco/plot_style.py`).

**Colab / Linux** usually has no TNR installed, so matplotlib falls back to **Liberation Serif** or **STIXGeneral** (still serif, not the default sans).

To get real Times New Roman on Colab, run once per runtime:

```bash
apt-get update -qq && apt-get install -y -qq fonts-liberation ttf-mscorefonts-installer
fc-cache -f
```

Optional: drop `*.ttf` files in this folder; they are registered automatically at plot time.
