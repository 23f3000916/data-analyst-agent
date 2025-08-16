import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import BytesIO
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

url_regex = re.compile(r'https?://[^\s]+')

def find_urls(text):
    return re.findall(url_regex, text)

def fetch_url_text(url, timeout=30):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "data-analyst-agent/1.0"})
    r.raise_for_status()
    return r.text

def read_html_tables(url_or_html):
    """Try to read HTML tables using pandas. url_or_html can be a URL or raw HTML."""
    try:
        if url_or_html.strip().startswith('<'):
            tables = pd.read_html(url_or_html)
        else:
            tables = pd.read_html(url_or_html)
        return tables
    except Exception:
        return []

def series_corr(a, b):
    return float(np.corrcoef(a, b)[0,1])

def make_scatter_with_regression(df, x_col, y_col, dotted_line=True, color_line='red', max_size_bytes=100000):
    """
    Returns a data URI `data:image/png;base64,...` for a scatterplot with a regression line.
    Attempts to reduce image bytes to under max_size_bytes by lowering DPI and converting to webp if needed.
    """
    x = df[x_col].astype(float)
    y = df[y_col].astype(float)

    # Fit linear regression
    coeffs = np.polyfit(x, y, 1)  # slope, intercept
    slope, intercept = coeffs[0], coeffs[1]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(x, y)
    # regression line
    xs = np.linspace(x.min(), x.max(), 200)
    ys = slope * xs + intercept
    linestyle = '--' if dotted_line else '-'
    ax.plot(xs, ys, linestyle, linewidth=1.5)
    ax.set_xlabel(str(x_col))
    ax.set_ylabel(str(y_col))
    ax.grid(True)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    png_data = buf.getvalue()

    # If too large, try reducing DPI and converting
    if len(png_data) > max_size_bytes:
        # Try reducing DPI
        for dpi in [80, 60, 40, 20]:
            buf = BytesIO()
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(x, y)
            ax.plot(xs, ys, linestyle, linewidth=1.5)
            ax.set_xlabel(str(x_col))
            ax.set_ylabel(str(y_col))
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=dpi)
            plt.close(fig)
            png_data = buf.getvalue()
            if len(png_data) <= max_size_bytes:
                break

    # If still too large, convert to WEBP (Pillow)
    if len(png_data) > max_size_bytes:
        try:
            im = Image.open(BytesIO(png_data)).convert('RGB')
            wb = BytesIO()
            im.save(wb, format='WEBP', quality=60)
            webp_data = wb.getvalue()
            if len(webp_data) < len(png_data):
                data_b64 = base64.b64encode(webp_data).decode('ascii')
                return f'data:image/webp;base64,{data_b64}', float(slope)
        except Exception:
            pass

    data_b64 = base64.b64encode(png_data).decode('ascii')
    return f'data:image/png;base64,{data_b64}', float(slope)

def compress_png_bytes(png_bytes, max_size=100000):
    if len(png_bytes) <= max_size:
        return png_bytes
    try:
        im = Image.open(BytesIO(png_bytes)).convert('RGB')
        wb = BytesIO()
        im.save(wb, format='WEBP', quality=60)
        webp = wb.getvalue()
        if len(webp) <= max_size:
            return webp
    except Exception:
        pass
    # fallback: return original
    return png_bytes
