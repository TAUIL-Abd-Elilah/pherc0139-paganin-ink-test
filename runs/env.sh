# Edit these, then: source runs/env.sh   (run everything from the repository root)
export CT_CACHE=${CT_CACHE:-cache}              # zfetch chunk cache (alignment boxes, scroll-centre slices)
export XSCAN_CACHE=${XSCAN_CACHE:-cache/xscan}  # render_crop chunk cache, ~1 GB per render
export PAG_OUT=${PAG_OUT:-out}                  # meshes, labels, renders and model maps per segment
export CT_CAP_GB=${CT_CAP_GB:-60}
export SEGS="20260317000000-w035_2026031718 20260302000000-w039_2026030210 20250831000000-w040_2025083102 20260108000000-w041_2026010816 20260115000000-w044_2026011522"
# models (src/models.py only)
export VILLA_SRC=${VILLA_SRC:-villa/vesuvius/src}                                  # ScrollPrize/villa checkout
export INK9UM_S42=${INK9UM_S42:-models/hybrid_3d2d-seed42/step-075000.pth}         # organisers' ink_9um release
export INK9UM_S43=${INK9UM_S43:-models/hybrid_3d2d-seed43/step-075000.pth}
export READER_V2=${READER_V2:-models/reader-v2-step040000.pth}                      # github.com/DomRusso2/reader-v2
export V8IN_DIR=${V8IN_DIR:-models/ink-8um-v8in}                                    # huggingface.co/YoussefMoNader/ink-8um-v8in
