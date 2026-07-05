import re, sys
from pypdf import PdfReader
src = "/refs/MPPI-DK_2603.05385_Accelerating-Sampling-Based-Control-Linear-Koopman.pdf"
r = PdfReader(src)
if r.is_encrypted:
    r.decrypt("")
print("pages:", len(r.pages))
txt = "\n".join((p.extract_text() or "") for p in r.pages)
for kw in ["pendulum", "swing", "invert", "balanc", "lifting dimension", "EDMD", "horizon"]:
    hits = [m.start() for m in re.finditer("(?i)"+kw, txt)]
    print(f"\n===== '{kw}'  ({len(hits)} hits) =====")
    for s in hits[:6]:
        print("...", txt[max(0, s-260):s+360].replace("\n", " "))
