# BandGap

A desktop GUI tool for extracting the **band gap from single STS spectra** (Omicron `.dat` format).
It locates the conduction band edge (Ec) and valence band edge (Ev) in each dI/dV spectrum and reports the band gap Eg, with three detection methods to compare.

Companion to [STS-map](https://github.com/RinoWu96/STS-map), which handles full line scans.

## Features

- Three band-edge detection methods:
  | Method | Idea |
  |---|---|
  | Adaptive Thresholding | Dynamic noise threshold on log(dI/dV) around the in-gap plateau |
  | Linear Extrapolation | Linear fits to the band onsets, extrapolated to the plateau level |
  | Slope Detection | First point where the log(dI/dV) slope exceeds a threshold |
- Browse a folder of `.dat` files with Previous / Next and a file list
- Linear, log or normalised Y axis; drag to adjust the plateau region
- Average mode: combine several spectra and compute the averaged band gap
- Batch process a whole folder to a TXT results file
- Export figures (SVG / PDF / PNG), save results to TXT, or copy them to the clipboard
- Chinese / English interface

## Download (Windows)

No Python needed: download `STS_line.exe` from the [Releases](https://github.com/RinoWu96/BandGap/releases) page and double-click it.

## Run from source

Requires Python 3.9+.

```bash
git clone https://github.com/RinoWu96/BandGap.git
cd BandGap
pip install -r requirements.txt
python STS_line.py
```

### Build the executable yourself

```bash
pip install pyinstaller
pyinstaller STS_line.spec
# output: dist/STS_line.exe
```

## Basic workflow

1. Click **Select .dat File** and pick a spectrum. The other `.dat` files in the same folder appear in the file list.
2. Check the Bias, Current and LIx column names; change them if they were not detected correctly.
3. Choose an analysis method and, if needed, adjust its parameters (defaults are a good start).
4. Click **Run Analysis**. Ec, Ev and Eg are shown in the results panel.
5. Use **Batch Process** to analyse every `.dat` file in a folder at once.

## Citation

If this tool helps your research, please cite:

```
Rino. BandGap: a GUI tool for band gap extraction from STS spectra. GitHub, 2026. https://github.com/RinoWu96/BandGap
```

## Contact

Rino
