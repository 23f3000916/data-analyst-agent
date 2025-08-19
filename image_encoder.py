# image_encoder.py
import base64
import io

def encode_plot_to_base64(fig):
    """
    Encode matplotlib figure to PNG base64 string under 100 KB.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.read()

    # Reduce DPI until under 100 KB
    while len(img_bytes) > 100_000:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        buf.seek(0)
        img_bytes = buf.read()

    base64_img = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_img}"
