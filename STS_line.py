#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于动态阈值与梯度判定的 STS 带隙提取 GUI
改进自 band_edge_core.py 核心算法
"""

import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("TkAgg")

# 统一配置 Matplotlib 字体为系统默认，解决打包后中文显示问题
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False 

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# 显式导入 SVG 和 PDF 后端，确保 PyInstaller 打包时能正确识别并包含这些模块
import matplotlib.backends.backend_svg
import matplotlib.backends.backend_pdf

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

# ----------------- 适应高 DPI 显示 (Windows) ----------------- #
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# ----------------- 数据读取 ----------------- #

def read_sts_single(path: Path):
    encodings = ['utf8', 'latin-1', 'cp1252']
    lines = []
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                lines = f.read().splitlines()
            if lines: break
        except Exception:
            continue
            
    if not lines:
        raise ValueError(f"Could not read file {path}")

    data_start = None
    for i, s in enumerate(lines):
        if s.strip() == "[DATA]":
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"No [DATA] section in {path}")

    # 数据从 [DATA] 后的第一个非空行开始，第一行即为表头
    data_lines = lines[data_start:]
    header_idx = 0
    for i, line in enumerate(data_lines):
        if line.strip():
            header_idx = i
            break
            
    data_str = "\n".join(data_lines[header_idx:])
    
    # 优先使用 Tab 分隔，因为列名可能包含空格
    try:
        df = pd.read_csv(StringIO(data_str), sep="\t")
        if len(df.columns) <= 1:
            df = pd.read_csv(StringIO(data_str), sep=None, engine='python')
    except Exception:
        df = pd.read_csv(StringIO(data_str), sep=None, engine='python')
        
    return df

# ----------------- 改进的核心算法 ----------------- #

def preprocess_sts_data(df, vcol, icol, lcol, smooth_window, shift_multiplier=0.0, v_center_min=None, v_center_max=None):
    V_raw = df[vcol].to_numpy(dtype=float)
    S_raw_orig = df[lcol].to_numpy(dtype=float)
    raw_I_orig = df[icol].to_numpy(dtype=float) if icol in df.columns else None

    # 1. 确定用于计算噪音标准差的范围 (Plateau)
    if v_center_min is None or v_center_max is None:
        # 临时排序并计算 log 以寻找平台区
        idx_temp = np.argsort(V_raw)
        V_temp, S_temp = V_raw[idx_temp], S_raw_orig[idx_temp]
        logS_temp = np.log10(np.abs(S_temp) + 1e-30)
        v_p_min, v_p_max = get_plateau_range(V_temp, logS_temp)
    else:
        v_p_min, v_p_max = v_center_min, v_center_max

    # 2. 计算噪音区标准差并确定 offset
    noise_mask = (V_raw >= v_p_min) & (V_raw <= v_p_max)
    sigma_noise = np.std(S_raw_orig[noise_mask]) if noise_mask.any() else np.std(S_raw_orig)
    offset = shift_multiplier * sigma_noise

    # 3. 应用平移
    S_shifted = S_raw_orig + offset
    
    # 4. 排序
    idx = np.argsort(V_raw)
    V, S_shifted = V_raw[idx], S_shifted[idx]
    raw_I = raw_I_orig[idx] if raw_I_orig is not None else None

    # 5. 直接取绝对值后取对数
    logS_raw = np.log10(np.abs(S_shifted))
    win = int(smooth_window)
    if win % 2 == 0:
        win += 1
    if len(logS_raw) > win:
        # 处理可能出现的 inf 或 nan
        finite_vals = logS_raw[np.isfinite(logS_raw)]
        if len(finite_vals) > 0:
            fill_min, fill_max = np.nanmin(finite_vals), np.nanmax(finite_vals)
        else:
            fill_min, fill_max = -30, 0
            
        logS_raw_clean = np.nan_to_num(logS_raw, nan=fill_min, posinf=fill_max, neginf=fill_min)
        logS_smooth = savgol_filter(logS_raw_clean, win, polyorder=2)
    else:
        logS_smooth = logS_raw
    I_smooth = 10 ** logS_smooth
    return V, I_smooth, logS_smooth, raw_I, S_shifted, logS_raw, offset

def get_plateau_range(V, logS_raw):
    vmin, vmax = float(V.min()), float(V.max())
    v_span = vmax - vmin
    v_abs_max = max(abs(vmin), abs(vmax))

    mid_mask = np.abs(V) < 0.25 * v_abs_max
    if mid_mask.sum() < 10:
        mid_mask = np.abs(V) < 0.35 * v_abs_max
    if mid_mask.sum() < 10:
        mid_mask = np.ones_like(V, dtype=bool)

    log_mid = logS_raw[mid_mask]
    V_mid = V[mid_mask]
    base_guess = float(np.median(log_mid))
    # 平移后基准线会变高，这里 0.3 的容差通常依然适用，因为它反映的是对数尺度的波动
    plateau_mask_mid = log_mid < (base_guess + 0.3)
    if np.any(plateau_mask_mid):
        v_plateau_min = float(V_mid[plateau_mask_mid].min())
        v_plateau_max = float(V_mid[plateau_mask_mid].max())
    else:
        v_plateau_min, v_plateau_max = -0.3, 0.3

    margin = 0.05 * v_abs_max
    v_center_min = v_plateau_min - margin
    v_center_max = v_plateau_max + margin
    v_center_min = max(v_center_min, vmin + 0.05 * v_span)
    v_center_max = min(v_center_max, vmax - 0.05 * v_span)
    if v_center_min >= v_center_max:
        v_center_min = vmin * 0.2
        v_center_max = vmax * 0.2
    return v_center_min, v_center_max

def find_edges_dynamic_core(V, logS_raw, threshold):
    Ev, Ec = np.nan, np.nan
    
    # --- 导带边缘 (Ec) ---
    mask_cb = (V > 0) & (logS_raw > threshold)
    if np.any(mask_cb):
        idx_cb_all = np.where(mask_cb)[0]
        idx_trigger = None
        for k in idx_cb_all:
            if k < len(logS_raw) - 9 and np.all(logS_raw[k:k+10] > threshold):
                idx_trigger = k
                break
        
        if idx_trigger is not None and idx_trigger > 0:
            v1, v2 = V[idx_trigger-1], V[idx_trigger]
            y1, y2 = logS_raw[idx_trigger-1], logS_raw[idx_trigger]
            if abs(y2 - y1) > 1e-15:
                Ec = v1 + (threshold - y1) * (v2 - v1) / (y2 - y1)
            else:
                Ec = V[idx_trigger]
        elif len(idx_cb_all) > 0:
            Ec = V[idx_cb_all[0]]

    # --- 价带边缘 (Ev) ---
    mask_vb = (V < 0) & (logS_raw > threshold)
    if np.any(mask_vb):
        idx_vb_all = np.where(mask_vb)[0]
        idx_trigger = None
        for k in sorted(idx_vb_all, reverse=True):
            if k >= 9 and np.all(logS_raw[k-9:k+1] > threshold):
                idx_trigger = k
                break
        
        if idx_trigger is not None and idx_trigger < len(logS_raw) - 1:
            v1, v2 = V[idx_trigger], V[idx_trigger+1]
            y1, y2 = logS_raw[idx_trigger], logS_raw[idx_trigger+1]
            if abs(y2 - y1) > 1e-15:
                Ev = v1 + (threshold - y1) * (v2 - v1) / (y2 - y1)
            else:
                Ev = V[idx_trigger]
        elif len(idx_vb_all) > 0:
            Ev = V[idx_vb_all[-1]]
            
    return Ev, Ec

def detect_band_edges_dynamic(
    df: pd.DataFrame,
    vcol: str = "Bias calc (V)",
    icol: str = "Current (A)",
    lcol: str = "LI Demod 1 X (A)",
    smooth_window: int = 11,
    sigma_level: float = 2.0,
    slope_percentile: float = 85,
    min_gap: float = 0.05,
    v_center_min: float = None,
    v_center_max: float = None,
    delta_E: float = 0.15,
    shift_multiplier: float = 0.0,
):
    V, I_smooth, logS_smooth, raw_I, S_shifted, logS_raw, offset = preprocess_sts_data(
        df, vcol, icol, lcol, smooth_window, shift_multiplier=shift_multiplier, v_center_min=v_center_min, v_center_max=v_center_max
    )
    
    if v_center_min is None or v_center_max is None:
        v_center_min, v_center_max = get_plateau_range(V, logS_raw)

    center_mask = (V >= v_center_min) & (V <= v_center_max)
    # 仅当完全没有选中点时才回退到自动检测
    if center_mask.sum() < 1:
        v_center_min, v_center_max = get_plateau_range(V, logS_raw)
        center_mask = (V >= v_center_min) & (V <= v_center_max)
    
    if np.any(center_mask):
        data_lin_raw = S_shifted[center_mask]
    else:
        data_lin_raw = S_shifted
        
    base_lin = float(np.mean(data_lin_raw))  # 改为使用平均值
    sigma_lin = float(np.std(data_lin_raw, ddof=1)) if len(data_lin_raw) > 1 else 0.0 # 统一使用标准差 (ddof=1)
    
    base = np.log10(base_lin + 1e-18)
    
    if sigma_level is not None and sigma_level > 0:
        thresh_lin = base_lin + sigma_level * sigma_lin
        sigma_log_contrib = np.log10(thresh_lin + 1e-18) - base
    else:
        sigma_log_contrib = 0.0

    left_high = np.percentile(logS_raw[V < v_center_min], 90) if (V < v_center_min).any() else base + 1.0
    right_high = np.percentile(logS_raw[V > v_center_max], 90) if (V > v_center_max).any() else base + 1.0
    span = max(0.5, min(left_high - base, right_high - base))
    
    delta_E_floor = 0.05 
    # 动态调整用于阈值判定的 delta_E_thr，根据 span 范围
    # 根据项目建议，使用 0.6 * span 动态系数
    delta_E_thr = max(delta_E_floor, min(0.6 * span, 1.2))
    
    # 只有当 sigma_level 为 None 或 <=0 时，才完全依赖 delta_E_thr
    if sigma_level is not None and sigma_level > 0:
        thr_add = sigma_log_contrib
    else:
        thr_add = delta_E_thr
        
    threshold = float(base + thr_add)

    dlog_dV = np.gradient(logS_raw, V)
    slope_thr = float(np.percentile(np.abs(dlog_dV), slope_percentile))

    # 第一步：自适应阈值判定初始带边
    Ev, Ec = find_edges_dynamic_core(V, logS_raw, threshold)

    def linear_fit_R2(x, y):
        if len(x) < 3: return None, None, None
        p = np.polyfit(x, y, 1)
        ss_res = np.sum((y - np.polyval(p, x))**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return p[0], p[1], float(R2)

    # 第二步：线性拟合精细化 (参考 band_edge_core.py)
    a_cb, b_cb, R2_cb = None, None, None
    cb_fit_V, cb_fit_logI = None, None
    if not np.isnan(Ec):
        # 寻找 Ec 后的拟合窗口
        cb_mask = (V >= Ec) & (V <= Ec + delta_E) & (V > 0)
        a_cb, b_cb, R2_cb = linear_fit_R2(V[cb_mask], logS_raw[cb_mask])
        if a_cb is not None and a_cb > 0:
            Ec = (base - b_cb) / a_cb
            cb_fit_V, cb_fit_logI = V[cb_mask], logS_raw[cb_mask]

    a_vb, b_vb, R2_vb = None, None, None
    vb_fit_V, vb_fit_logI = None, None
    if not np.isnan(Ev):
        # 寻找 Ev 前的拟合窗口
        vb_mask = (V >= Ev - delta_E) & (V <= Ev) & (V < 0)
        a_vb, b_vb, R2_vb = linear_fit_R2(V[vb_mask], logS_raw[vb_mask])
        if a_vb is not None and a_vb < 0:
            Ev = (base - b_vb) / a_vb
            vb_fit_V, vb_fit_logI = V[vb_mask], logS_raw[vb_mask]

    Eg = Ec - Ev if not (np.isnan(Ec) or np.isnan(Ev)) else np.nan
    if not np.isnan(Eg) and Eg < min_gap: Ev, Ec, Eg = np.nan, np.nan, np.nan

    res = {
        "Eg": 0.0 if np.isnan(Eg) else float(Eg),
        "Ev": Ev, "Ec": Ec, "V": V, "I": I_smooth, "logS": logS_smooth, "raw_I": raw_I, "logS_raw": logS_raw,
        "floor_log": base, "sigma": sigma_lin, "sigma_level": sigma_level, 
        "threshold": threshold, "y_2sigma_log": threshold, "slope_thr": slope_thr,
        "v_center_min": float(v_center_min), "v_center_max": float(v_center_max),
        "method": "Adaptive Thresholding", "offset": offset,
        "note": "Success" if not np.isnan(Eg) else "Gap not clearly detected",
        "R2_vb": R2_vb, "R2_cb": R2_cb
    }
    
    if a_vb is not None:
        res.update({"a_vb": a_vb, "b_vb": b_vb, "vb_fit_V": vb_fit_V, "vb_fit_logI": vb_fit_logI})
    if a_cb is not None:
        res.update({"a_cb": a_cb, "b_cb": b_cb, "cb_fit_V": cb_fit_V, "cb_fit_logI": cb_fit_logI})
        
    return res

def extract_gap_log_plateau(
    df: pd.DataFrame,
    vcol: str = "Bias calc (V)",
    icol: str = "Current (A)",
    lcol: str = "LI Demod 1 X (A)",
    v_center_min: float = -0.5,
    v_center_max: float = 0.3,
    delta_E: float = 0.15,
    min_gap: float = 0.1,
    smooth_window: int = 11,
    sigma_level: float = 2.0,
    shift_multiplier: float = 0.0,
):
    V, I_smooth, logS_smooth, raw_I, S_shifted, logS_raw, offset = preprocess_sts_data(
        df, vcol, icol, lcol, smooth_window, shift_multiplier=shift_multiplier, v_center_min=v_center_min, v_center_max=v_center_max
    )
    
    center_mask = (V >= v_center_min) & (V <= v_center_max)
    # 仅当完全没有选中点时才回退到自动检测，允许用户选择较小的范围
    if center_mask.sum() < 1:
        v_center_min_auto, v_center_max_auto = get_plateau_range(V, logS_raw)
        center_mask = (V >= v_center_min_auto) & (V <= v_center_max_auto)

    if center_mask.sum() > 0:
        floor_lin = float(np.mean(S_shifted[center_mask]))
        sigma_lin = float(np.std(S_shifted[center_mask], ddof=1)) if len(S_shifted[center_mask]) > 1 else 0.0
        # 基准线直接在 log 空间统计（中位数），而不是先在线性空间平均再取 log。
        # 线性平均会被正负噪音相互抵消拉向 0，导致 log10(mean) 系统性偏低，
        # 落在 log 坐标图上噪音散点云的下方而非中间。
        floor_log = float(np.median(logS_raw[center_mask]))
        sigma_log = float(np.std(logS_raw[center_mask], ddof=1)) if len(logS_raw[center_mask]) > 1 else 0.0
    else:
        floor_lin = float(np.mean(S_shifted))
        sigma_lin = float(np.std(S_shifted))
        floor_log = float(np.median(logS_raw))
        sigma_log = float(np.std(logS_raw))

    y_thresh_log = floor_log + sigma_level * sigma_log
    y_floor = floor_log

    # 基础返回字典，包含统计信息
    base_res = {
        "Eg": 0.0, "Ev": np.nan, "Ec": np.nan,
        "V": V, "I": I_smooth, "logS": logS_smooth, "raw_I": raw_I, "logS_raw": logS_raw,
        "floor_log": floor_log, "sigma": sigma_lin, "sigma_log": sigma_log, "sigma_level": sigma_level, "y_2sigma_log": y_thresh_log,
        "method": "Linear Extrapolation", "offset": offset
    }

    def find_crossing_index(mask, is_cb=True):
        if not mask.any(): return None
        indices = np.where(mask)[0]
        if is_cb:
            for i in indices:
                # 检查连续10个点大于阈值
                if i < len(logS_raw) - 9 and np.all(logS_raw[i:i+10] > y_thresh_log):
                    return i
        else:
            # 价带向负压方向搜索
            for i in sorted(indices, reverse=True):
                # 检查连续10个点（向负压方向延伸）大于阈值
                if i >= 9 and np.all(logS_raw[i-9:i+1] > y_thresh_log):
                    return i
        return None

    idx_ev = find_crossing_index(V < 0, is_cb=False)
    idx_ec = find_crossing_index(V > 0, is_cb=True)

    if idx_ev is None or idx_ec is None:
        base_res["note"] = f"No points above {sigma_level}sigma"
        return base_res

    # --- 确定拟合起始点 ---
    idx_ev_start = min(idx_ev + 1, len(V) - 1)
    if V[idx_ev_start] >= 0: idx_ev_start = idx_ev 
    idx_ec_start = max(idx_ec - 1, 0)
    if V[idx_ec_start] <= 0: idx_ec_start = idx_ec 

    v_ev_start = V[idx_ev_start]
    v_ec_start = V[idx_ec_start]

    def linear_fit(x, y):
        if len(x) < 3: return None, None, None
        p = np.polyfit(x, y, 1)
        R2 = 1.0 - np.sum((y - np.polyval(p, x))**2) / np.sum((y - np.mean(y))**2) if np.std(y) > 0 else 1.0
        return p[0], p[1], float(R2)

    vb_mask = (V >= v_ev_start - delta_E) & (V <= v_ev_start) & (V < 0)
    a_vb, b_vb, R2_vb = linear_fit(V[vb_mask], logS_raw[vb_mask]) if vb_mask.sum() >= 3 else (None, None, None)
    if a_vb is not None and a_vb >= 0: a_vb = b_vb = R2_vb = None

    cb_mask = (V >= v_ec_start) & (V <= v_ec_start + delta_E) & (V > 0)
    a_cb, b_cb, R2_cb = linear_fit(V[cb_mask], logS_raw[cb_mask]) if cb_mask.sum() >= 3 else (None, None, None)
    if a_cb is not None and a_cb <= 0: a_cb = b_cb = R2_cb = None

    if a_vb is None or a_cb is None:
        base_res["note"] = "Fit failed"
        return base_res

    Evbm, Ecbm = (y_floor - b_vb) / a_vb, (y_floor - b_cb) / a_cb
    Eg = Ecbm - Evbm
    if Eg < min_gap: Eg = 0.0

    # 填充成功后的结果
    base_res.update({
        "Eg": Eg, "Ev": Evbm, "Ec": Ecbm,
        "Ev_2sigma": V[idx_ev], "Ec_2sigma": V[idx_ec],
        "a_vb": a_vb, "b_vb": b_vb, "a_cb": a_cb, "b_cb": b_cb, "R2_vb": R2_vb, "R2_cb": R2_cb,
        "vb_fit_V": V[vb_mask], "vb_fit_logI": logS_raw[vb_mask], "cb_fit_V": V[cb_mask], "cb_fit_logI": logS_raw[cb_mask],
        "note": "Success" if Eg > 0 else f"Eg={Eg:.3f} < min_gap"
    })
    return base_res

def detect_band_edges_slope(
    df: pd.DataFrame,
    vcol: str = "Bias calc (V)",
    icol: str = "Current (A)",
    lcol: str = "LI Demod 1 X (A)",
    smooth_window: int = 11,
    slope_threshold: float = 0.5,
    min_gap: float = 0.05,
    v_center_min: float = None,
    v_center_max: float = None,
    # --- 内部参数：不在 GUI 暴露，保持向后兼容 ---
    run_len: int = 8,
    run_len_cb: int = 5,
    fit_window: float = 0.15,
    baseline_k: float = 0.0,
    amp_k: float = 2.0,
    r2_min: float = 0.90,
    # --- VB 曲率兜底参数 ---
    curv_k: float = 2.0,
    curv_percentile: float = 95,
    run_len_curv: int = 3,
    vb_q: float = 0.80,
    shift_multiplier: float = 0.0,
):
    """    斜率触发 + 线性外推（线性 dI/dV 空间，使用 LI X），并在 VB 侧加入“曲率兜底”以抑制软尾。

    ✅ 修正点（针对你反馈的 Ec 过严）：
    - CB 侧触发不再使用幅度门槛 amp_mask（因为 CB 通常无“软尾”问题，幅度门槛会过度推迟触发点/导致找不到）。
    - CB 侧连续点数单独使用 run_len_cb（默认 5），比 VB 的 run_len（默认 8）更宽松。

    主流程（线性外推）：
    1) 线性 dI/dV 平滑后求一阶导 g=d(dI/dV)/dV；
    2) 在平台(带隙)区估计 g 的噪声，阈值 g_thr=max(slope_threshold*sigma_g, noise_g)
       - slope_threshold 解释为“噪声倍数”（无量纲），避免不同数据量纲导致阈值不可比。
    3) VB：连续 run_len 点满足 g<-g_thr 且 S>amp_thr（抑制软尾）；
       CB：连续 run_len_cb 点满足 g>g_thr（不加幅度门槛）。
    4) 在触发点附近拟合/外推得到 Ev/Ec 候选。

    VB 曲率兜底：若 VB 外推不可用，则用二阶导 curv=d²(dI/dV)/dV² 辅助定位主态 onset。
    """

    V, I_smooth_logspace, logS_smooth, raw_I, S_shifted, logS_raw, offset = preprocess_sts_data(
        df, vcol, icol, lcol, smooth_window, shift_multiplier=shift_multiplier, v_center_min=v_center_min, v_center_max=v_center_max
    )

    S_lin = S_shifted - offset

    win = int(smooth_window)
    if win % 2 == 0:
        win += 1
    if len(S_lin) > win:
        S_smooth = savgol_filter(S_lin, win, polyorder=2)
    else:
        S_smooth = S_lin.copy()

    g = np.gradient(S_smooth, V)
    curv = np.gradient(g, V)

    if v_center_min is None or v_center_max is None:
        v_center_min, v_center_max = get_plateau_range(V, logS_raw)
    plateau_mask = (V >= float(v_center_min)) & (V <= float(v_center_max))
    if plateau_mask.sum() < 10:
        plateau_mask = np.abs(V) < 0.3 * np.max(np.abs(V))

    sigma_g = float(np.std(g[plateau_mask], ddof=1)) if plateau_mask.sum() > 1 else float(np.std(g))
    noise_g = float(np.percentile(np.abs(g[plateau_mask]), 80)) if plateau_mask.any() else float(np.percentile(np.abs(g), 80))
    sigma_g = max(sigma_g, 1e-30)
    noise_g = max(noise_g, 1e-30)
    g_thr = float(max(slope_threshold * sigma_g, noise_g))

    sigma_curv = float(np.std(curv[plateau_mask], ddof=1)) if plateau_mask.sum() > 1 else float(np.std(curv))
    noise_curv = float(np.percentile(np.abs(curv[plateau_mask]), curv_percentile)) if plateau_mask.any() else float(np.percentile(np.abs(curv), curv_percentile))
    sigma_curv = max(sigma_curv, 1e-30)
    noise_curv = max(noise_curv, 1e-30)
    curv_thr = float(max(curv_k * sigma_curv, noise_curv))

    floor_lin = float(np.median(S_smooth[plateau_mask])) if plateau_mask.any() else float(np.median(S_smooth))
    sigma_lin = float(np.std(S_smooth[plateau_mask], ddof=1)) if plateau_mask.sum() > 1 else float(np.std(S_smooth))
    sigma_lin = max(sigma_lin, 1e-30)
    y_floor = float(floor_lin + baseline_k * sigma_lin)
    amp_thr = float(y_floor + amp_k * sigma_lin)

    def _first_run(mask: np.ndarray, run: int, from_left: bool = True):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None
        splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        if from_left:
            for seg in splits:
                if len(seg) >= run:
                    return int(seg[0])
        else:
            for seg in splits[::-1]:
                if len(seg) >= run:
                    return int(seg[-run])
        return None

    # VB：保留幅度门槛（抑制软尾）
    amp_mask = S_smooth > amp_thr
    vb_mask = (V < 0) & (g < -g_thr) & amp_mask
    idx_vb0 = _first_run(vb_mask, run_len, from_left=False)

    # CB：取消幅度门槛（更宽松）
    cb_mask = (V > 0) & (g > g_thr)
    idx_cb0 = _first_run(cb_mask, run_len_cb, from_left=True)

    Ev_trigger = float(V[idx_vb0]) if idx_vb0 is not None else np.nan
    Ec_trigger = float(V[idx_cb0]) if idx_cb0 is not None else np.nan

    Ev = Ec = np.nan
    Ev_candidate = Ec_candidate = np.nan
    Ev_curv = np.nan

    # 仍保留旧版的“基于局部斜率外推”做法（它不会计算真实 R²，因此 R2_* 维持 1.0）
    a_vb = b_vb = R2_vb = None
    a_cb = b_cb = R2_cb = None
    vb_fit_V = vb_fit_y = None
    cb_fit_V = cb_fit_y = None

    # VB 外推：在触发点及其内部（更负方向）进行拟合
    if idx_vb0 is not None:
        v_trig, g_trig = V[idx_vb0], g[idx_vb0]
        # 窗口向更负方向延伸 (交点后)
        near_mask = (V >= v_trig - fit_window) & (V <= v_trig) & (V < 0)
        # 宽松一点的过滤：只要斜率方向对即可，或者斜率在一定范围内
        valid_mask = near_mask & (g < 0)
        if valid_mask.sum() >= 3:
            a_vb = float(np.mean(g[valid_mask]))
            avg_V = float(np.mean(V[valid_mask]))
            avg_S = float(np.mean(S_smooth[valid_mask]))
            b_vb = float(avg_S - a_vb * avg_V)
        else:
            a_vb = float(g_trig)
            b_vb = float(S_smooth[idx_vb0] - a_vb * v_trig)
        R2_vb = 1.0
        if a_vb < 0:
            Ev_candidate = float((y_floor - b_vb) / a_vb)
            Ev = Ev_candidate
            v_start = min(Ev, v_trig - fit_window) - 0.05
            v_end = max(Ev, v_trig) + 0.05
            vb_fit_V = np.array([v_start, v_end])
            vb_fit_y = a_vb * vb_fit_V + b_vb

    # CB 外推：在触发点及其内部（更正方向）进行拟合
    if idx_cb0 is not None:
        v_trig, g_trig = V[idx_cb0], g[idx_cb0]
        # 窗口向更正方向延伸 (交点后)
        near_mask = (V >= v_trig) & (V <= v_trig + fit_window) & (V > 0)
        valid_mask = near_mask & (g > 0)
        if valid_mask.sum() >= 3:
            a_cb = float(np.mean(g[valid_mask]))
            avg_V = float(np.mean(V[valid_mask]))
            avg_S = float(np.mean(S_smooth[valid_mask]))
            b_cb = float(avg_S - a_cb * avg_V)
        else:
            a_cb = float(g_trig)
            b_cb = float(S_smooth[idx_cb0] - a_cb * v_trig)
        R2_cb = 1.0
        if a_cb > 0:
            Ec_candidate = float((y_floor - b_cb) / a_cb)
            Ec = Ec_candidate
            v_start = min(Ec, v_trig) - 0.05
            v_end = max(Ec, v_trig + fit_window) + 0.05
            cb_fit_V = np.array([v_start, v_end])
            cb_fit_y = a_cb * cb_fit_V + b_cb

    # VB 曲率兜底（当 VB 外推没有给出 Ev 时才启用）
    if np.isnan(Ev):
        neg_mask = V < 0
        q_thr = float(np.percentile(S_smooth[neg_mask], vb_q * 100.0)) if np.any(neg_mask) else amp_thr
        amp_thr2 = float(max(amp_thr, q_thr))

        vb_curv_mask = (V < 0) & (np.abs(curv) > curv_thr) & (g < 0) & (S_smooth > amp_thr2)
        idx_vb_curv = _first_run(vb_curv_mask, run_len_curv, from_left=False)
        if idx_vb_curv is None:
            cand = np.where((V < 0) & (g < 0) & (S_smooth > amp_thr2))[0]
            if cand.size > 0:
                j = cand[np.argmax(np.abs(curv[cand]))]
                if np.abs(curv[j]) > curv_thr:
                    idx_vb_curv = int(j)
        if idx_vb_curv is not None:
            Ev_curv = float(V[idx_vb_curv])
            Ev = Ev_curv

    Eg = 0.0
    note = "Gap not found"
    if not (np.isnan(Ev) or np.isnan(Ec)) and Ev < Ec:
        Eg = float(Ec - Ev)
        if Eg < min_gap:
            Eg, Ev, Ec = 0.0, np.nan, np.nan
            note = f"Eg<{min_gap}"
        else:
            note = "Success"
    else:
        reasons = []
        if np.isnan(Ec_trigger):
            reasons.append("CB trigger not found (too strict)")
        if np.isnan(Ev_trigger) and np.isnan(Ev_curv):
            reasons.append("VB trigger/curv not found")
        note = "; ".join(reasons) if reasons else note

    return {
        "Eg": Eg,
        "Ev": float(Ev) if not np.isnan(Ev) else np.nan,
        "Ec": float(Ec) if not np.isnan(Ec) else np.nan,
        "Ev_candidate": float(Ev_candidate) if not np.isnan(Ev_candidate) else np.nan,
        "Ec_candidate": float(Ec_candidate) if not np.isnan(Ec_candidate) else np.nan,
        "Ev_curv": float(Ev_curv) if not np.isnan(Ev_curv) else np.nan,
        "V": V,
        "I": S_smooth + offset,
        "logS": logS_smooth,
        "raw_I": raw_I,
        "logS_raw": logS_raw,
        "S_raw_lin": S_lin,
        "S_smooth_lin": S_smooth,
        "slopes": g,
        "curv": curv,
        "threshold": g_thr,
        "curv_thr": curv_thr,
        "sigma_g": sigma_g,
        "noise_g": noise_g,
        "sigma_curv": sigma_curv,
        "noise_curv": noise_curv,
        "floor_lin": floor_lin,
        "sigma_lin": sigma_lin,
        "y_floor": y_floor,
        "amp_thr": amp_thr,
        "Ev_trigger": Ev_trigger,
        "Ec_trigger": Ec_trigger,
        "a_vb": a_vb,
        "b_vb": b_vb,
        "R2_vb": R2_vb,
        "a_cb": a_cb,
        "b_cb": b_cb,
        "R2_cb": R2_cb,
        "vb_fit_V_lin": vb_fit_V,
        "vb_fit_y_lin": vb_fit_y,
        "cb_fit_V_lin": cb_fit_V,
        "cb_fit_y_lin": cb_fit_y,
        "v_center_min": float(v_center_min),
        "v_center_max": float(v_center_max),
        "method": "Slope Detection",
        "offset": offset,
        "note": note,
        "fit_space": "linear",
        "params": {
            "run_len": run_len,
            "run_len_cb": run_len_cb,
            "fit_window": fit_window,
            "baseline_k": baseline_k,
            "amp_k": amp_k,
            "r2_min": r2_min,
            "curv_k": curv_k,
            "curv_percentile": curv_percentile,
            "run_len_curv": run_len_curv,
            "vb_q": vb_q,
        },
    }

# ----------------- 国际化配置 ----------------- #

TRANSLATIONS = {
    "English": {
        "title": "STS Band Gap Analyzer (Multi-Method)",
        "choose_file": "Select .dat File",
        "batch_process": "Batch Process (to TXT)",
        "start_analysis": "Start Analysis",
        "method_frame": "Analysis Method",
        "param_frame": "Parameter Settings",
        "sigma_level": "Sigma Level:",
        "slope_percentile": "Slope %:",
        "smooth_window": "Smooth Win:",
        "min_gap": "Min Gap (eV):",
        "delta_e": "ΔE (eV):",
        "gap_min": "Gap_min (V):",
        "gap_max": "Gap_max (V):",
        "slope_threshold": "Slope Threshold:",
        "s_shift": "Shift Multiplier:",
        "options_frame": "Options",
        "bias_col": "Bias Column:",
        "curr_col": "Current Column:",
        "lix_col": "LI Demod Column:",
        "sigma_multiplier": "Sigma Multiplier:",
        "scan_err": "Failed to scan file columns: {err}",
        "show_annotations": "Show Annotations",
        "y_axis": "Y-Axis:",
        "linear": "Linear",
        "log": "Log",
        "normalize": "Normalize",
        "language": "Language:",
        "msg_finish": "Finished",
        "msg_batch_done": "Batch processing completed!\nTotal {} files processed.\nResults saved to: {}",
        "msg_warn": "Warning",
        "msg_no_dat": "No .dat files found in this folder.",
        "msg_save_title": "Save Batch Results",
        "msg_error": "Analysis Error",
        "overview": "Overview",
        "threshold": "Threshold",
        "floor": "Floor",
        "floor_sigma": "Floor+2σ",
        "gap_min_line": "Gap Min",
        "gap_max_line": "Gap Max",
        "vb_fit": "VB Fit",
        "cb_fit": "CB Fit",
        "ev": "Ev",
        "ec": "Ec",
        "raw_data": "Raw Data",
        "smooth_data": "Smooth Data",
        "prev_file": "Prev File",
        "next_file": "Next File",
        "file_list": "File List",
        "results_frame": "Analysis Results",
        "save_results": "Save Analysis Results",
        "save_fig1": "Save left View",
        "save_fig2": "Save right View",
        "save_success": "Figure saved: {path}",
        "average_mode": "Average Mode",
        "copy_clipboard": "Copy to Clipboard",
        "msg_copied": "Data copied to clipboard!",
        "methods": {
            "Adaptive Thresholding": "Adaptive Thresholding",
            "Linear Extrapolation": "Linear Extrapolation",
            "Slope Detection": "Slope Detection"
        }
    },
    "中文": {
        "title": "STS 带隙分析仪 (多方法整合)",
        "choose_file": "选择 .dat 文件",
        "batch_process": "批量处理 (生成TXT)",
        "start_analysis": "开始分析",
        "method_frame": "分析方法",
        "param_frame": "参数设置",
        "sigma_level": "Sigma 水平:",
        "slope_percentile": "斜率分位数 %:",
        "smooth_window": "平滑窗口:",
        "min_gap": "最小带隙 (eV):",
        "delta_e": "ΔE (eV):",
        "gap_min": "带隙下限 (V):",
        "gap_max": "带隙上限 (V):",
        "slope_threshold": "斜率阈值:",
        "s_shift": "平移倍率 (x Sigma):",
        "options_frame": "选项设置",
        "bias_col": "电压列 (Bias):",
        "curr_col": "电流列 (Current):",
        "lix_col": "LI Demod列:",
        "sigma_multiplier": "Sigma 倍率:",
        "scan_err": "扫描文件列名失败: {err}",
        "show_annotations": "显示注释",
        "y_axis": "Y轴标尺:",
        "linear": "线性",
        "log": "对数",
        "normalize": "归一",
        "language": "语言:",
        "msg_finish": "完成",
        "msg_batch_done": "批量处理已完成！\n共处理 {} 个文件。\n结果已保存至: {}",
        "msg_warn": "警告",
        "msg_no_dat": "该文件夹内没有找到 .dat 文件",
        "msg_save_title": "保存批量处理结果",
        "msg_error": "分析出错",
        "overview": "全景图",
        "threshold": "阈值",
        "floor": "基准线",
        "floor_sigma": "基准+2σ",
        "gap_min_line": "区域下限",
        "gap_max_line": "区域上限",
        "vb_fit": "价带拟合",
        "cb_fit": "导带拟合",
        "ev": "价带顶",
        "ec": "导带底",
        "raw_data": "原始数据",
        "smooth_data": "平滑数据",
        "prev_file": "上一个",
        "next_file": "下一个",
        "file_list": "文件列表",
        "results_frame": "分析结果",
        "save_results": "保存分析结果",
        "save_fig1": "保存左图",
        "save_fig2": "保存右图",
        "save_success": "图片已保存: {path}",
        "average_mode": "求平均模式",
        "copy_clipboard": "复制到剪贴板",
        "msg_copied": "数据已复制到剪贴板！",
        "methods": {
            "Adaptive Thresholding": "自适应阈值法",
            "Linear Extrapolation": "线性外推法",
            "Slope Detection": "斜率判定法"
        }
    }
}

# ----------------- GUI 部分 ----------------- #

class LogGapGUI:
    def __init__(self, master):
        self.master = master
        
        # 设置全局默认字体，解决打包后文字可能无法显示的问题
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Microsoft YaHei", size=9)
        master.option_add("*Font", default_font)
        
        self.lang_var = tk.StringVar(value="中文")
        
        self.file_path = tk.StringVar()
        self.method_var = tk.StringVar(value="Adaptive Thresholding")
        self.method_keys = ["Adaptive Thresholding", "Linear Extrapolation", "Slope Detection"]
        
        self.sigma_level = tk.StringVar(value="2.0")
        self.slope_percentile = tk.StringVar(value="85")
        self.smooth_window = tk.StringVar(value="11")
        self.min_gap = tk.StringVar(value="0.05")
        self.delta_E = tk.StringVar(value="0.15")
        self.v_center_min = tk.StringVar(value="-0.3")
        self.v_center_max = tk.StringVar(value="0.3")
        self.s_shift = tk.StringVar(value="0.0")
        
        # 列名选择变量
        self.bias_col_var = tk.StringVar(value="Bias calc (V)")
        self.curr_col_var = tk.StringVar(value="Current (A)")
        self.lix_col_var = tk.StringVar(value="LI Demod 1 X (A)")
        
        # New parameters
        self.slope_threshold = tk.StringVar(value="0.5")
        self.sigma_multiplier = tk.StringVar(value="5.0")
        
        self.show_annotations = tk.BooleanVar(value=True)
        self.average_mode = tk.BooleanVar(value=False)
        self.y_scale = tk.StringVar(value="linear")

        self.last_averaged_data = None # Store last averaged result for clipboard
        self.current_results = []      # Store all results from last run_analysis

        self.dragging = False
        self.current_line = None
        self.gap_min_line = None
        self.gap_max_line = None
        
        self.file_list = []
        self.current_file_idx = -1

        # --- 全局布局: 主水平 PanedWindow --- #
        self.main_paned = tk.PanedWindow(master, orient="horizontal", sashrelief="raised", sashwidth=4)
        self.main_paned.pack(fill="both", expand=True)

        # --- 左侧面板: 控制与列表 (使用 Canvas 配合 Scrollbar 以适配小屏幕高度) --- #
        self.left_container = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_container, width=420) # 给定初始宽度

        self.left_canvas = tk.Canvas(self.left_container, borderwidth=0, highlightthickness=0)
        self.left_scrollbar = ttk.Scrollbar(self.left_container, orient="vertical", command=self.left_canvas.yview)
        self.left_scrollable_frame = ttk.Frame(self.left_canvas)

        self.left_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        )
        # 移除全局鼠标滚轮绑定，使左侧界面固定，仅允许通过滚动条手动滑动
        # def _on_mousewheel(event):
        #     self.left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        # self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.left_canvas.create_window((0, 0), window=self.left_scrollable_frame, anchor="nw")
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

        self.left_canvas.pack(side="left", fill="both", expand=True)
        self.left_scrollbar.pack(side="right", fill="y")

        # 将所有原本在 master 下的组件放入 left_scrollable_frame
        ctrl_parent = self.left_scrollable_frame

        # --- Top: File & Method --- #
        top = ttk.Frame(ctrl_parent)
        top.pack(fill="x", padx=10, pady=5)

        # Language Selection
        lang_frame = ttk.Frame(top)
        lang_frame.grid(row=0, column=0, columnspan=2, sticky="w")
        self.lbl_lang = ttk.Label(lang_frame, text="语言:")
        self.lbl_lang.pack(side="left")
        self.lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var, values=["English", "中文"], state="readonly", width=8)
        self.lang_combo.pack(side="left", padx=5)
        self.lang_combo.bind("<<ComboboxSelected>>", lambda e: self.update_ui_text())

        self.btn_choose = ttk.Button(top, text="选择 .dat 文件", command=self.choose_file)
        self.btn_choose.grid(row=1, column=0, sticky="w")
        ttk.Label(top, textvariable=self.file_path, width=40).grid(row=1, column=1, padx=5, sticky="w")

        # Navigation Buttons
        nav_frame = ttk.Frame(top)
        nav_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky="w")
        self.btn_prev = ttk.Button(nav_frame, text="上一个", width=10, command=lambda: self.change_file(-1))
        self.btn_prev.pack(side="left", padx=2)
        self.btn_next = ttk.Button(nav_frame, text="下一个", width=10, command=lambda: self.change_file(1))
        self.btn_next.pack(side="left", padx=2)

        self.method_frame = ttk.LabelFrame(top, text="分析方法")
        self.method_frame.grid(row=3, column=0, columnspan=4, pady=5, sticky="w")
        
        self.method_combo = ttk.Combobox(self.method_frame, textvariable=self.method_var, state="readonly", width=35)
        self.method_combo.pack(side="left", padx=5, pady=5)
        self.method_combo.bind("<<ComboboxSelected>>", lambda e: self.on_method_change())

        # --- Parameters --- #
        self.param_frame = ttk.LabelFrame(top, text="参数设置")
        self.param_frame.grid(row=4, column=0, columnspan=4, pady=5, sticky="w")

        # Row 0
        self.lbl_sigma = ttk.Label(self.param_frame, text="Sigma Level:")
        self.lbl_sigma.grid(row=0, column=0, sticky="e")
        self.ent_sigma = ttk.Entry(self.param_frame, textvariable=self.sigma_level, width=8)
        self.ent_sigma.grid(row=0, column=1, padx=5, pady=2)

        self.lbl_slope = ttk.Label(self.param_frame, text="Slope %:")
        self.lbl_slope.grid(row=0, column=2, sticky="e")
        self.ent_slope = ttk.Entry(self.param_frame, textvariable=self.slope_percentile, width=8)
        self.ent_slope.grid(row=0, column=3, padx=5, pady=2)

        # Row 1
        self.lbl_smooth = ttk.Label(self.param_frame, text="Smooth Win:")
        self.lbl_smooth.grid(row=1, column=0, sticky="e")
        self.ent_smooth = ttk.Entry(self.param_frame, textvariable=self.smooth_window, width=8)
        self.ent_smooth.grid(row=1, column=1, padx=5, pady=2)

        self.lbl_min_gap = ttk.Label(self.param_frame, text="Min Gap (eV):")
        self.lbl_min_gap.grid(row=1, column=2, sticky="e")
        self.ent_min_gap = ttk.Entry(self.param_frame, textvariable=self.min_gap, width=8)
        self.ent_min_gap.grid(row=1, column=3, padx=5, pady=2)

        # Row 2
        self.lbl_delta_e = ttk.Label(self.param_frame, text="ΔE (eV):")
        self.lbl_delta_e.grid(row=2, column=0, sticky="e")
        self.ent_delta_e = ttk.Entry(self.param_frame, textvariable=self.delta_E, width=8)
        self.ent_delta_e.grid(row=2, column=1, padx=5, pady=2)

        self.lbl_slope_thresh = ttk.Label(self.param_frame, text="Slope Threshold:")
        self.lbl_slope_thresh.grid(row=2, column=2, sticky="e")
        self.ent_slope_thresh = ttk.Entry(self.param_frame, textvariable=self.slope_threshold, width=8)
        self.ent_slope_thresh.grid(row=2, column=3, padx=5, pady=2)

        # Row 3 (Plateau)
        self.lbl_vmin = ttk.Label(self.param_frame, text="Gap_min (V):")
        self.lbl_vmin.grid(row=3, column=0, sticky="e")
        self.ent_vmin = ttk.Entry(self.param_frame, textvariable=self.v_center_min, width=8)
        self.ent_vmin.grid(row=3, column=1, padx=5, pady=2)

        self.lbl_vmax = ttk.Label(self.param_frame, text="Gap_max (V):")
        self.lbl_vmax.grid(row=3, column=2, sticky="e")
        self.ent_vmax = ttk.Entry(self.param_frame, textvariable=self.v_center_max, width=8)
        self.ent_vmax.grid(row=3, column=3, padx=5, pady=2)

        # Row 4 (Column Selection 1)
        self.lbl_bias_col = ttk.Label(self.param_frame, text="Bias Col:")
        self.lbl_bias_col.grid(row=4, column=0, sticky="e")
        self.combo_bias = ttk.Combobox(self.param_frame, textvariable=self.bias_col_var, state="readonly", width=12)
        self.combo_bias.grid(row=4, column=1, padx=5, pady=2)

        self.lbl_curr_col = ttk.Label(self.param_frame, text="Curr Col:")
        self.lbl_curr_col.grid(row=4, column=2, sticky="e")
        self.combo_curr = ttk.Combobox(self.param_frame, textvariable=self.curr_col_var, state="readonly", width=12)
        self.combo_curr.grid(row=4, column=3, padx=5, pady=2)

        # Row 5 (Column Selection 2)
        self.lbl_lix_col = ttk.Label(self.param_frame, text="LIx Col:")
        self.lbl_lix_col.grid(row=5, column=0, sticky="e")
        self.combo_lix = ttk.Combobox(self.param_frame, textvariable=self.lix_col_var, state="readonly", width=12)
        self.combo_lix.grid(row=5, column=1, padx=5, pady=2)

        self.lbl_sigma_multiplier = ttk.Label(self.param_frame, text="Sigma Multiplier:")
        self.lbl_sigma_multiplier.grid(row=5, column=2, sticky="e")
        self.ent_sigma_multiplier = ttk.Entry(self.param_frame, textvariable=self.sigma_multiplier, width=8)
        self.ent_sigma_multiplier.grid(row=5, column=3, padx=5, pady=2)

        # Row 6
        self.lbl_s_shift = ttk.Label(self.param_frame, text="Shift Mult:")
        self.lbl_s_shift.grid(row=6, column=0, sticky="e")
        self.ent_s_shift = ttk.Entry(self.param_frame, textvariable=self.s_shift, width=8)
        self.ent_s_shift.grid(row=6, column=1, padx=5, pady=2)

        # --- Options --- #
        self.opt_frame = ttk.LabelFrame(top, text="选项")
        self.opt_frame.grid(row=5, column=0, columnspan=4, sticky="w", pady=5)
        self.chk_anno = ttk.Checkbutton(self.opt_frame, text="显示注释", variable=self.show_annotations)
        self.chk_anno.pack(side="left", padx=5)
        
        self.chk_avg = ttk.Checkbutton(self.opt_frame, text="求平均模式", variable=self.average_mode, command=self.run_analysis)
        self.chk_avg.pack(side="left", padx=5)
        
        self.lbl_y_axis = ttk.Label(self.opt_frame, text="Y轴:")
        self.lbl_y_axis.pack(side="left", padx=(10,0))
        self.rad_linear = ttk.Radiobutton(self.opt_frame, text="Linear", value="linear", variable=self.y_scale, command=self.run_analysis)
        self.rad_linear.pack(side="left", padx=5)
        self.rad_log = ttk.Radiobutton(self.opt_frame, text="Log", value="log", variable=self.y_scale, command=self.run_analysis)
        self.rad_log.pack(side="left", padx=5)
        self.rad_norm = ttk.Radiobutton(self.opt_frame, text="Normalize", value="normalize", variable=self.y_scale, command=self.run_analysis)
        self.rad_norm.pack(side="left", padx=5)

        # Action Buttons Frame
        btn_frame = ttk.Frame(ctrl_parent)
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        # Row 1 of buttons
        btn_row1 = ttk.Frame(btn_frame)
        btn_row1.pack(fill="x")
        self.btn_run = ttk.Button(btn_row1, text="开始分析", command=self.run_analysis)
        self.btn_run.pack(side="left", padx=2, pady=2)
        
        self.btn_batch = ttk.Button(btn_row1, text="批量处理", command=self.batch_process_to_txt)
        self.btn_batch.pack(side="left", padx=2, pady=2)
        
        self.btn_save_res = ttk.Button(btn_row1, text="保存分析结果", command=self.save_current_results)
        self.btn_save_res.pack(side="left", padx=2, pady=2)

        # Row 2 of buttons
        btn_row2 = ttk.Frame(btn_frame)
        btn_row2.pack(fill="x")
        self.btn_save_fig1 = ttk.Button(btn_row2, text="保存左图", command=lambda: self.save_figure_manually(1))
        self.btn_save_fig1.pack(side="left", padx=2, pady=2)
        
        self.btn_save_fig2 = ttk.Button(btn_row2, text="保存右图", command=lambda: self.save_figure_manually(2))
        self.btn_save_fig2.pack(side="left", padx=2, pady=2)

        self.btn_copy = ttk.Button(btn_row2, text="复制到剪贴板", command=self.copy_to_clipboard)
        self.btn_copy.pack(side="left", padx=2, pady=2)

        # File Listbox
        list_frame = ttk.LabelFrame(ctrl_parent, text="文件列表")
        list_frame.pack(fill="x", padx=10, pady=5)
        self.list_frame = list_frame
        
        self.lb_files = tk.Listbox(list_frame, height=8, font=default_font, selectmode="extended")
        self.lb_files.pack(side="left", fill="both", expand=True)
        self.lb_files.bind("<<ListboxSelect>>", self.on_list_select)
        
        lb_sb = ttk.Scrollbar(list_frame, command=self.lb_files.yview)
        lb_sb.pack(side="right", fill="y")
        self.lb_files.config(yscrollcommand=lb_sb.set)

        # Text Output
        info_frame = ttk.LabelFrame(ctrl_parent, text="分析结果")
        self.info_frame = info_frame
        info_frame.pack(fill="x", padx=10, pady=5)
        self.info_text = tk.Text(info_frame, height=8, wrap="word", font=("Microsoft YaHei", 9))
        self.info_text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        sb.pack(side="right", fill="y")
        self.info_text.config(yscrollcommand=sb.set)

        # --- 右侧面板: 图形显示 --- #
        self.right_container = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_container, stretch="always")

        # 使用垂直 PanedWindow 堆叠两个图，或者水平排列？
        # 考虑到宽屏，水平排列两个 4:3 比例的图在高度上利用率更高
        self.plot_paned = tk.PanedWindow(self.right_container, orient="horizontal", sashrelief="raised", sashwidth=4)
        self.plot_paned.pack(fill="both", expand=True)

        self.fig1 = Figure(figsize=(6, 4.5), dpi=100)
        self.fig2 = Figure(figsize=(6, 4.5), dpi=100)
        
        self.plot_container1 = tk.Frame(self.plot_paned, bg="white")
        self.plot_container2 = tk.Frame(self.plot_paned, bg="white")
        
        self.plot_paned.add(self.plot_container1, stretch="always")
        self.plot_paned.add(self.plot_container2, stretch="always")
        
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.plot_container1)
        self.canvas1_widget = self.canvas1.get_tk_widget()
        
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.plot_container2)
        self.canvas2_widget = self.canvas2.get_tk_widget()
        
        # 核心：监听容器尺寸变化，动态调整画布以保持指定比例
        self.plot_ratio = 4/3
        self.pane2_hidden = False

        def make_resize_handler(cv_widget):
            def on_resize(event):
                w, h = event.width, event.height
                if w <= 0 or h <= 0: return
                target_ratio = self.plot_ratio
                current_ratio = w / h
                if current_ratio > target_ratio:
                    new_h = h
                    new_w = h * target_ratio
                else:
                    new_w = w
                    new_h = w / target_ratio
                cv_widget.place(relx=0.5, rely=0.5, anchor="center", width=new_w, height=new_h)
            return on_resize

        self.plot_container1.bind("<Configure>", make_resize_handler(self.canvas1_widget))
        self.plot_container2.bind("<Configure>", make_resize_handler(self.canvas2_widget))
        
        # 事件绑定 (这里需要决定绑定到哪个 Canvas，或者两个都绑定)
        # 考虑到拖拽基准线主要在细节图(图2)，我们绑定到 canvas2
        self.canvas2.mpl_connect('pick_event', self.on_pick)
        self.canvas2.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas2.mpl_connect('button_release_event', self.on_release)

        self.update_ui_text()
        self.update_param_visibility()

    def update_ui_text(self):
        lang = self.lang_var.get()
        t = TRANSLATIONS[lang]
        
        self.master.title(t["title"])
        self.btn_choose.config(text=t["choose_file"])
        self.btn_batch.config(text=t["batch_process"])
        self.btn_save_res.config(text=t["save_results"])
        self.btn_run.config(text=t["start_analysis"])
        self.btn_save_fig1.config(text=t["save_fig1"])
        self.btn_save_fig2.config(text=t["save_fig2"])
        self.method_frame.config(text=t["method_frame"])
        self.param_frame.config(text=t["param_frame"])
        
        self.lbl_lang.config(text=t["language"])
        self.lbl_sigma.config(text=t["sigma_level"])
        self.lbl_slope.config(text=t["slope_percentile"])
        self.lbl_smooth.config(text=t["smooth_window"])
        self.lbl_min_gap.config(text=t["min_gap"])
        self.lbl_delta_e.config(text=t["delta_e"])
        self.lbl_vmin.config(text=t["gap_min"])
        self.lbl_vmax.config(text=t["gap_max"])
        self.lbl_slope_thresh.config(text=t["slope_threshold"])
        self.lbl_s_shift.config(text=t["s_shift"])
        
        self.lbl_bias_col.config(text=t["bias_col"])
        self.lbl_curr_col.config(text=t["curr_col"])
        self.lbl_lix_col.config(text=t["lix_col"])
        self.lbl_sigma_multiplier.config(text=t["sigma_multiplier"])
        
        self.btn_prev.config(text=t["prev_file"])
        self.btn_next.config(text=t["next_file"])
        self.list_frame.config(text=t["file_list"])
        self.info_frame.config(text=t["results_frame"])
        
        self.opt_frame.config(text=t["options_frame"])
        self.chk_anno.config(text=t["show_annotations"])
        self.chk_avg.config(text=t["average_mode"])
        self.btn_copy.config(text=t["copy_clipboard"])
        self.lbl_y_axis.config(text=t["y_axis"])
        self.rad_linear.config(text=t["linear"])
        self.rad_log.config(text=t["log"])
        self.rad_norm.config(text=t["normalize"])
        
        # Update Method Names in Combobox
        current_method = self.method_var.get()
        # Find which key corresponds to the current display name
        lang_keys = ["English", "中文"]
        other_lang = lang_keys[0] if lang == "中文" else lang_keys[1]
        
        # Map current display name back to key if necessary
        key_found = current_method
        for k, v in TRANSLATIONS[other_lang]["methods"].items():
            if v == current_method:
                key_found = k
                break
        
        new_values = [t["methods"][k] for k in self.method_keys]
        self.method_combo.config(values=new_values)
        self.method_var.set(t["methods"].get(key_found, new_values[0]))

    def get_method_key(self):
        lang = self.lang_var.get()
        current_display = self.method_var.get()
        for k, v in TRANSLATIONS[lang]["methods"].items():
            if v == current_display:
                return k
        return self.method_keys[0]

    def on_method_change(self):
        self.update_param_visibility()

    def update_param_visibility(self):
        method_key = self.get_method_key()
        # 重置所有参数输入框为可用状态
        all_ents = [
            self.ent_sigma, self.ent_slope, self.ent_delta_e, 
            self.ent_vmin, self.ent_vmax, self.ent_slope_thresh,
            self.ent_sigma_multiplier
        ]
        for ent in all_ents:
            ent.config(state="normal")
        
        if method_key == "Adaptive Thresholding":
            self.ent_slope_thresh.config(state="disabled")
        elif method_key == "Linear Extrapolation":
            self.ent_slope.config(state="disabled")
            self.ent_slope_thresh.config(state="disabled")
        elif method_key == "Slope Detection":
            self.ent_sigma.config(state="disabled")
            self.ent_slope.config(state="disabled")
            self.ent_delta_e.config(state="normal")
            # 允许用户拖拽/输入带隙范围，用于斜率法的噪声/基线估计
            self.ent_vmin.config(state="normal")
            self.ent_vmax.config(state="normal")

    def batch_process_to_txt(self):
        lang = self.lang_var.get()
        t = TRANSLATIONS[lang]
        
        # 1. 选择输入文件夹
        folder_path = filedialog.askdirectory(title=t["choose_file"])
        if not folder_path:
            return
        
        p = Path(folder_path)
        files = list(p.glob("*.dat"))
        if not files:
            messagebox.showwarning(t["msg_warn"], t["msg_no_dat"])
            return

        # 2. 选择输出保存位置 (txt 格式)
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile="batch_results.txt",
            title=t["msg_save_title"]
        )
        if not save_path:
            return

        # 3. 获取当前 GUI 设定的参数
        method_key = self.get_method_key()
        sig = float(self.sigma_level.get())
        win = int(self.smooth_window.get())
        m_gap = float(self.min_gap.get())
        d_delta_e = float(self.delta_E.get())
        s_thresh = float(self.slope_threshold.get())
        s_shift_val = float(self.s_shift.get())
        
        # 获取用户选择的列名
        b_col = self.bias_col_var.get()
        c_col = self.curr_col_var.get()
        l_col = self.lix_col_var.get()

        results = []
        
        # 4. 开始循环处理
        for f in files:
            try:
                df = read_sts_single(f)
                
                # 构造基础参数
                base_args = {
                    "df": df,
                    "vcol": b_col,
                    "icol": c_col,
                    "lcol": l_col,
                    "smooth_window": win,
                    "min_gap": m_gap
                }

                # 根据项目建议，批量处理时应确保每个文件独立执行自动范围识别 (get_plateau_range)，
                # 避免共用 GUI 上的固定参数导致偏差。因此此处不传递 GUI 的 v_center_min/max。
                if method_key == "Adaptive Thresholding":
                    res = detect_band_edges_dynamic(
                        **base_args, 
                        sigma_level=sig, 
                        v_center_min=None,
                        v_center_max=None,
                        delta_E=d_delta_e,
                        shift_multiplier=s_shift_val
                    )
                elif method_key == "Linear Extrapolation":
                    res = extract_gap_log_plateau(
                        **base_args, 
                        v_center_min=None, 
                        v_center_max=None, 
                        delta_E=d_delta_e, 
                        sigma_level=sig,
                        shift_multiplier=s_shift_val
                    )
                elif method_key == "Slope Detection":
                    res = detect_band_edges_slope(
                        **base_args, 
                        slope_threshold=s_thresh,
                        v_center_min=None,
                        v_center_max=None,
                        fit_window=d_delta_e,
                        shift_multiplier=s_shift_val
                    )
                else:
                    continue
                
                # 收集数据行
                results.append(f"{f.name}\t{res['Eg']:.6f}\t{res['Ev']:.6f}\t{res['Ec']:.6f}")
            except Exception as e:
                print(f"File {f.name} failed: {e}")

        # 5. 写入 TXT 文件
        with open(save_path, "w", encoding="utf-8") as out_file:
            # 写入表头
            out_file.write("FileName\tEg(eV)\tEv(V)\tEc(V)\n")
            # 写入结果
            out_file.write("\n".join(results))

        messagebox.showinfo(t["msg_finish"], t["msg_batch_done"].format(len(results), save_path))

    def save_current_results(self):
        """将当前显示的分析结果保存到 TXT 文件"""
        lang = self.lang_var.get()
        t = TRANSLATIONS[lang]
        
        results = []
        if hasattr(self, 'current_results') and self.current_results:
            if self.average_mode.get() and self.last_averaged_data:
                results = [self.last_averaged_data]
            else:
                results = self.current_results
        
        if not results:
            messagebox.showwarning(t["msg_warn"], "No results to save.")
            return

        # 弹出保存对话框
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile="analysis_results.txt",
            title=t["msg_save_title"]
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", encoding="utf-8") as out_file:
                out_file.write("FileName\tEg(eV)\tEv(V)\tEc(V)\n")
                for res in results:
                    out_file.write(f"{res['filename']}\t{res['Eg']:.6f}\t{res['Ev']:.6f}\t{res['Ec']:.6f}\n")
            
            messagebox.showinfo(t["msg_finish"], f"Results saved to: {save_path}")
        except Exception as e:
            messagebox.showerror(t["msg_error"], f"Save failed: {str(e)}")


    def save_figure_manually(self, fig_num):
        """手动保存图1或图2为矢量图或PDF"""
        lang = self.lang_var.get()
        t = TRANSLATIONS[lang]
        
        target_fig = self.fig1 if fig_num == 1 else self.fig2
        
        if target_fig is None or not target_fig.axes:
            messagebox.showinfo(t["msg_finish"], "No data to save. Please run analysis first.")
            return
            
        # 弹出保存对话框
        file_path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg"), ("PDF files", "*.pdf"), ("PNG files", "*.png")],
            title=t[f"save_fig{fig_num}"]
        )
        
        if not file_path:
            return
            
        try:
            # 尝试使用 tight 布局保存
            target_fig.savefig(file_path, bbox_inches="tight")
            messagebox.showinfo(t["msg_finish"], t["save_success"].format(path=file_path))
        except Exception as e:
            try:
                # 如果 tight 失败，尝试普通保存
                target_fig.savefig(file_path)
                messagebox.showinfo(t["msg_finish"], t["save_success"].format(path=file_path) + "\n(Note: Saved without tight layout)")
            except Exception as e2:
                messagebox.showerror(t["msg_error"], f"Save failed: {str(e2)}\nOriginal error: {str(e)}")

    def update_file_list(self, current_path):
        p = Path(current_path).parent
        self.file_list = sorted(list(p.glob("*.dat")), key=lambda x: x.name)
        
        self.lb_files.delete(0, tk.END)
        for f in self.file_list:
            self.lb_files.insert(tk.END, f.name)
            
        try:
            self.current_file_idx = self.file_list.index(Path(current_path))
            self.lb_files.selection_clear(0, tk.END)
            self.lb_files.selection_set(self.current_file_idx)
            self.lb_files.see(self.current_file_idx)
        except ValueError:
            self.current_file_idx = -1
        
        # 扫描列名
        self.scan_columns(current_path)

    def scan_columns(self, file_path):
        """扫描文件以获取列名并更新下拉列表"""
        try:
            df = read_sts_single(Path(file_path))
            cols = list(df.columns)
            
            # 更新下拉列表
            for combo in [self.combo_bias, self.combo_curr, self.combo_lix]:
                combo['values'] = cols
            
            # 自动匹配列名
            def auto_set(var, patterns):
                for c in cols:
                    if any(p.lower() in c.lower() for p in patterns):
                        var.set(c)
                        break
            
            auto_set(self.bias_col_var, ["bias", "volt"])
            auto_set(self.curr_col_var, ["current", "curr"])
            auto_set(self.lix_col_var, ["LI Demod 1 X", "LI Demod", "Lock-in"])
            
        except Exception as e:
            print(f"Error scanning columns: {e}")

    def change_file(self, delta):
        if not self.file_list: return
        new_idx = self.current_file_idx + delta
        if 0 <= new_idx < len(self.file_list):
            self.current_file_idx = new_idx
            new_path = self.file_list[new_idx]
            self.file_path.set(str(new_path))
            
            self.lb_files.selection_clear(0, tk.END)
            self.lb_files.selection_set(new_idx)
            self.lb_files.see(new_idx)
            
            self.run_analysis()

    def on_list_select(self, event):
        selection = self.lb_files.curselection()
        if selection:
            # 始终以选中的第一个文件作为“当前文件”参考
            idx = selection[0]
            self.current_file_idx = idx
            new_path = self.file_list[idx]
            self.file_path.set(str(new_path))
            # 执行分析（run_analysis 内部会处理多选）
            self.run_analysis()

    def auto_set_plateau(self, path):
        try:
            df = read_sts_single(Path(path))
            # 自动设置阈值时也需要列名，这里先尝试用当前选择或默认值
            V, _, _, _, _, logS_raw, _ = preprocess_sts_data(
                df, self.bias_col_var.get(), self.curr_col_var.get(), self.lix_col_var.get(), 
                int(self.smooth_window.get()), shift_multiplier=float(self.s_shift.get())
            )
            v_min, v_max = get_plateau_range(V, logS_raw)
            self.v_center_min.set(f"{v_min:.3f}")
            self.v_center_max.set(f"{v_max:.3f}")
        except: pass

    def choose_file(self):
        path = filedialog.askopenfilename(filetypes=[("Omicron .dat", "*.dat"), ("All files", "*.*")])
        if path:
            self.file_path.set(path)
            self.update_file_list(path)
            self.auto_set_plateau(path)
            self.run_analysis() # 选择后直接运行一次分析

    def run_analysis(self):
        # 获取所有选中的索引
        selection = self.lb_files.curselection()
        if not selection:
            if not self.file_path.get(): return
            # 如果 Listbox 没选中（例如初次加载），尝试使用当前索引
            if self.current_file_idx >= 0:
                selection = [self.current_file_idx]
            else:
                return

        results = []
        lang = self.lang_var.get()
        t = TRANSLATIONS[lang]

        try:
            for idx in selection:
                fpath = self.file_list[idx]
                df = read_sts_single(Path(fpath))
                method_key = self.get_method_key()
                
                common_params = {
                    "df": df,
                    "vcol": self.bias_col_var.get(),
                    "icol": self.curr_col_var.get(),
                    "lcol": self.lix_col_var.get(),
                    "smooth_window": int(self.smooth_window.get()),
                    "min_gap": float(self.min_gap.get())
                }

                if method_key == "Adaptive Thresholding":
                    res = detect_band_edges_dynamic(
                        **common_params, 
                        sigma_level=float(self.sigma_level.get()), 
                        slope_percentile=float(self.slope_percentile.get()),
                        v_center_min=float(self.v_center_min.get()),
                        v_center_max=float(self.v_center_max.get()),
                        delta_E=float(self.delta_E.get()),
                        shift_multiplier=float(self.s_shift.get())
                    )
                elif method_key == "Linear Extrapolation":
                    res = extract_gap_log_plateau(
                        **common_params, 
                        v_center_min=float(self.v_center_min.get()), 
                        v_center_max=float(self.v_center_max.get()), 
                        delta_E=float(self.delta_E.get()), 
                        sigma_level=float(self.sigma_level.get()),
                        shift_multiplier=float(self.s_shift.get())
                    )
                elif method_key == "Slope Detection":
                    res = detect_band_edges_slope(
                        **common_params,
                        slope_threshold=float(self.slope_threshold.get()),
                        v_center_min=float(self.v_center_min.get()),
                        v_center_max=float(self.v_center_max.get()),
                        fit_window=float(self.delta_E.get()),
                        shift_multiplier=float(self.s_shift.get())
                    )
                
                res["filename"] = Path(fpath).name
                results.append(res)
            
            self.current_results = results # 保存所有结果

            if len(results) == 1:
                self.plot_result(results[0])
                self.show_info(results[0])
                self.last_averaged_data = results[0]
            else:
                if self.average_mode.get():
                    self.calculate_and_plot_average(results)
                else:
                    self.plot_multi_results(results)
                    self.last_averaged_data = None
                # 多选时，信息框显示摘要
                self.show_multi_info(results)
                
        except Exception as e:
            messagebox.showerror(t["msg_error"], str(e))

    def plot_result(self, res):
        """将结果绘制到两个独立的 Figure 上"""
        # 恢复双图布局和比例
        if self.pane2_hidden:
            self.plot_paned.add(self.plot_container2, stretch="always")
            self.pane2_hidden = False
        
        if self.plot_ratio != 4/3:
            self.plot_ratio = 4/3
            # 触发 resize 以更新画布
            self.plot_container1.event_generate("<Configure>")
            self.plot_container2.event_generate("<Configure>")

        self.fig1.clf()
        self.fig2.clf()
        
        ax1 = self.fig1.add_subplot(111)
        ax2 = self.fig2.add_subplot(111)

        V, logS = res["V"], res["logS"]
        logS_raw = res.get("logS_raw")
        offset = res.get("offset", 0.0)
        method_name = res["method"] # This is the key or internal name
        lang = self.lang_var.get()
        t = TRANSLATIONS[lang]
        
        display_method = t["methods"].get(method_name, method_name)

        # Ax1: Linear or Log or Normalize
        y_mode = self.y_scale.get()
        if y_mode == "log":
            if logS_raw is not None:
                ax1.scatter(V, logS_raw, s=5, color="gray", alpha=0.3, label=f"log10(dI/dV) ({t['raw_data']})")
            ax1.plot(V, logS, color="black", linewidth=1, marker='o', markersize=3, label=f"log10(dI/dV) ({t['smooth_data']})")
            ax1.set_ylabel("log10(dI/dV)")
        elif y_mode == "normalize":
            # gN = (dI/dV) / (I/V)
            # 使用 Soft Regularization: 分母 I -> I + sign(I)*lambda, lambda = 5*sigma_I
            didv_smooth = res["I"] - offset
            raw_I = res["raw_I"]
            if raw_I is not None:
                # 1. 计算 sigma_I (电流噪声标准差)，优先在 gap 区域计算
                v_min, v_max = res.get("v_center_min"), res.get("v_center_max")
                if v_min is not None and v_max is not None:
                    gap_mask = (V >= v_min) & (V <= v_max)
                    sigma_I = np.nanstd(raw_I[gap_mask]) if gap_mask.any() else np.nanstd(raw_I)
                else:
                    sigma_I = np.nanstd(raw_I)
                
                lambd = float(self.sigma_multiplier.get()) * sigma_I
                
                # 2. 定义正则化 gN 计算函数
                # gN = (dI/dV) / [(I + sign(I)*lambda) / V] = (dI/dV * V) / (I + sign(I)*lambda)
                def calc_gn_soft(didv, I_vals, V_vals, lb):
                    # I_reg = I + sign(I)*lambda
                    # 使用 np.where 确保 sign(0) 时也应用 lambda
                    sign_I = np.where(I_vals >= 0, 1.0, -1.0)
                    I_reg = I_vals + sign_I * lb
                    
                    # 为了处理 V=0，我们使用 (didv * V) / I_reg
                    # 这样当 V=0 时，结果自然为 0，且避免了除以 V 的发散
                    gn = (didv * V_vals) / (I_reg + 1e-21)
                    return gn

                gN_smooth = calc_gn_soft(didv_smooth, raw_I, V, lambd)
                
                if logS_raw is not None:
                    didv_raw = 10**logS_raw - offset
                    gN_raw = calc_gn_soft(didv_raw, raw_I, V, lambd)
                    ax1.scatter(V, gN_raw, s=5, color="gray", alpha=0.3, label=f"gN ({t['raw_data']})")
                
                ax1.plot(V, gN_smooth, color="black", linewidth=1, marker='o', markersize=3, label=f"gN ({t['smooth_data']})")
                ax1.set_ylabel("gN (Soft Regularized)")
            else:
                ax1.text(0.5, 0.5, "Current column (I) missing", transform=ax1.transAxes, ha='center')
        else:
            # 线性显示时，减去 offset 以还原物理数值
            if logS_raw is not None:
                ax1.scatter(V, 10**logS_raw - offset, s=5, color="gray", alpha=0.3, label=f"dI/dV ({t['raw_data']})")
            ax1.plot(V, res["I"] - offset, color="black", linewidth=1, marker='o', markersize=3, label=f"dI/dV ({t['smooth_data']})")
            ax1.set_ylabel("dI/dV")
        
        if res["raw_I"] is not None:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(V, res["raw_I"], color="red", alpha=0.3, label="I-V")
            ax1_twin.set_ylabel("Current (A)")

        # 在全景图 (ax1) 中也展示 Ev 和 Ec
        if "Ev" in res and res["Ev"] is not None and not np.isnan(float(res["Ev"])):
            ax1.axvline(res["Ev"], color="red", linestyle=":", alpha=0.6)
        if "Ec" in res and res["Ec"] is not None and not np.isnan(float(res["Ec"])):
            ax1.axvline(res["Ec"], color="blue", linestyle=":", alpha=0.6)
        if res.get("Eg", 0) > 0 and res["Ev"] is not None and res["Ec"] is not None:
            if not np.isnan(float(res["Ev"])) and not np.isnan(float(res["Ec"])):
                ax1.axvspan(res["Ev"], res["Ec"], color="green", alpha=0.05)

        # Ax2: Details view
        if method_name == "Slope Detection":
            # 斜率法：右图显示线性 dI/dV（log 前），并叠加拟合与基线交点
            S_raw_lin = res.get("S_raw_lin", None)
            S_smooth_lin = res.get("S_smooth_lin", None)
            if S_raw_lin is not None:
                ax2.plot(V, S_raw_lin, color="gray", alpha=0.3, label=t["raw_data"])
            if S_smooth_lin is not None:
                ax2.plot(V, S_smooth_lin, color="black", alpha=0.8, label=t["smooth_data"])
            ax2.set_ylabel("dI/dV")

            # 基线
            y_floor = res.get("y_floor", None)
            if y_floor is not None and not np.isnan(float(y_floor)):
                ax2.axhline(y_floor, color="orange", linestyle="--", alpha=0.6, label="Baseline")

            # 幅度门槛（可选展示）
            amp_thr = res.get("amp_thr", None)
            if amp_thr is not None and not np.isnan(float(amp_thr)):
                ax2.axhline(amp_thr, color="orange", linestyle=":", alpha=0.35, label="Amp thr")

            # 拟合线（线性空间）
            if res.get('a_vb') is not None and res.get('vb_fit_V_lin') is not None:
                xv = res['vb_fit_V_lin']
                ax2.plot(xv, res['a_vb']*xv + res['b_vb'], color="red", linewidth=2, label="VB Fit")
            if res.get('a_cb') is not None and res.get('cb_fit_V_lin') is not None:
                xc = res['cb_fit_V_lin']
                ax2.plot(xc, res['a_cb']*xc + res['b_cb'], color="blue", linewidth=2, label="CB Fit")

            # 触发点
            if res.get('Ev_trigger') is not None and not np.isnan(float(res.get('Ev_trigger'))):
                ax2.axvline(res['Ev_trigger'], color="red", linestyle=":", alpha=0.4)
            if res.get('Ec_trigger') is not None and not np.isnan(float(res.get('Ec_trigger'))):
                ax2.axvline(res['Ec_trigger'], color="blue", linestyle=":", alpha=0.4)

            # 候选交点与最终交点（最终交点通过 R² 门槛）
            if y_floor is not None and not np.isnan(float(y_floor)):
                if res.get('Ev_candidate') is not None and not np.isnan(float(res.get('Ev_candidate'))):
                    ax2.plot([res['Ev_candidate']], [y_floor], marker='x', color='red', markersize=6, label='Ev cand')
                if res.get('Ec_candidate') is not None and not np.isnan(float(res.get('Ec_candidate'))):
                    ax2.plot([res['Ec_candidate']], [y_floor], marker='x', color='blue', markersize=6, label='Ec cand')
                if res.get('Ev') is not None and not np.isnan(float(res.get('Ev'))):
                    ax2.plot([res['Ev']], [y_floor], marker='o', color='red', markersize=5, label='Ev (R2 ok)')
                if res.get('Ec') is not None and not np.isnan(float(res.get('Ec'))):
                    ax2.plot([res['Ec']], [y_floor], marker='o', color='blue', markersize=5, label='Ec (R2 ok)')
                if res.get('Ev_curv') is not None and not np.isnan(float(res.get('Ev_curv'))):
                    ax2.axvline(res['Ev_curv'], color='purple', linestyle='--', alpha=0.6, label='Ev(curv)')

        else:
            # 其它方法：右图显示 log(dI/dV)
            if logS_raw is not None:
                ax2.plot(V, logS_raw, color="gray", alpha=0.3, label=t["raw_data"])
                ax2.plot(V, logS, color="black", alpha=0.7, label=t["smooth_data"])

        # Base/Threshold
        if method_name != "Slope Detection":
            if "threshold" in res and res["threshold"] is not None and not np.isnan(float(res["threshold"])):
                ax2.axhline(res["threshold"], color="orange", linestyle="--", label=t["threshold"])
        if "floor_log" in res and res["floor_log"] is not None:
            ax2.axhline(res["floor_log"], color="gray", linestyle="--", label=t["floor"])
            if "y_2sigma_log" in res and res["y_2sigma_log" ] is not None:
                sig_lv = res.get('sigma_level', 2.0)
                ax2.axhline(res["y_2sigma_log"], color="orange", linestyle=":", label=f"{t['floor']}+{sig_lv}σ")
            elif "sigma_log" in res and res["sigma_log"] is not None:
                sig_lv = res.get('sigma_level', 2.0)
                ax2.axhline(res["floor_log"] + sig_lv*res["sigma_log"], color="orange", linestyle=":", label=f"{t['floor']}+{sig_lv}σ")

        # Plateau range (Interactive)
        v_min = float(self.v_center_min.get())
        v_max = float(self.v_center_max.get())
        self.gap_min_line = ax2.axvline(v_min, color="magenta", linestyle="--", picker=5, label=t["gap_min_line"])
        self.gap_max_line = ax2.axvline(v_max, color="cyan", linestyle="--", picker=5, label=t["gap_max_line"])

        # Fit lines for Supplement
        if "vb_fit_V" in res and res["vb_fit_V"] is not None:
            ax2.plot(res["vb_fit_V"], res["a_vb"]*res["vb_fit_V"] + res["b_vb"], "r-", linewidth=2, label=t["vb_fit"])
        if "cb_fit_V" in res and res["cb_fit_V"] is not None:
            ax2.plot(res["cb_fit_V"], res["a_cb"]*res["cb_fit_V"] + res["b_cb"], "b-", linewidth=2, label=t["cb_fit"])

        # Edges
        if "Ev" in res and res["Ev"] is not None and not np.isnan(float(res["Ev"])):
            ax2.axvline(res["Ev"], color="red", label=f"{t['ev']}:{res['Ev']:.3f}")
        if "Ec" in res and res["Ec"] is not None and not np.isnan(float(res["Ec"])):
            ax2.axvline(res["Ec"], color="blue", label=f"{t['ec']}:{res['Ec']:.3f}")

        if res.get("Eg", 0) > 0 and res["Ev"] is not None and res["Ec"] is not None:
            if not np.isnan(float(res["Ev"])) and not np.isnan(float(res["Ec"])):
                ax2.axvspan(res["Ev"], res["Ec"], color="green", alpha=0.1)

        ax1.set_title(t["overview"])
        ax2.set_title(f"{display_method} (Eg={res['Eg']:.3f} eV)")
        if self.show_annotations.get(): 
            ax1.legend(fontsize=7)
            ax2.legend(fontsize=7)
        
        try:
            self.fig1.tight_layout()
            self.fig2.tight_layout()
        except:
            pass
        self.canvas1.draw()
        self.canvas2.draw()

    def plot_multi_results(self, results_list):
        """绘制多条曲线对比，带 Y 轴偏移 (Waterfall plot)"""
        # 切换到单图大画幅布局，并调整比例为较长的 Y 轴 (例如 1:1.2)
        if not self.pane2_hidden:
            self.plot_paned.forget(self.plot_container2)
            self.pane2_hidden = True
        
        # 调整比例，使 Y 轴方向更长 (比值 < 1 表示高大于宽)
        new_ratio = 1 / 1.1 
        if self.plot_ratio != new_ratio:
            self.plot_ratio = new_ratio
            self.plot_container1.event_generate("<Configure>")

        self.fig1.clf()
        
        ax1 = self.fig1.add_subplot(111)
        
        lang = self.lang_var.get()
        t = TRANSLATIONS[lang]
        y_mode = self.y_scale.get()
        
        plot_data = []
        y_mins, y_maxs = [], []

        for res in results_list:
            V = res["V"]
            offset = res.get("offset", 0.0)
            
            if y_mode == "log":
                y = res["logS"]
                ylabel = "log10(dI/dV)"
            elif y_mode == "normalize":
                didv_smooth = res["I"] - offset
                raw_I = res["raw_I"]
                if raw_I is not None:
                    # 简化 gN 计算
                    v_min, v_max = res.get("v_center_min"), res.get("v_center_max")
                    if v_min is not None and v_max is not None:
                        gap_mask = (V >= v_min) & (V <= v_max)
                        sigma_I = np.nanstd(raw_I[gap_mask]) if gap_mask.any() else np.nanstd(raw_I)
                    else:
                        sigma_I = np.nanstd(raw_I)
                    lambd = float(self.sigma_multiplier.get()) * sigma_I
                    sign_I = np.where(raw_I >= 0, 1.0, -1.0)
                    I_reg = raw_I + sign_I * lambd
                    y = (didv_smooth * V) / (I_reg + 1e-21)
                    ylabel = "gN (Soft Regularized)"
                else:
                    y = np.zeros_like(V)
                    ylabel = "gN (Missing I)"
            else:
                y = res["I"] - offset
                ylabel = "dI/dV"
            
            plot_data.append({
                "V": V, 
                "y": y, 
                "name": res["filename"], 
                "Ev": res.get("Ev"), 
                "Ec": res.get("Ec")
            })
            y_mins.append(np.nanmin(y))
            y_maxs.append(np.nanmax(y))

        if not plot_data: return

        # 计算自动偏移量
        global_min = min(y_mins)
        global_max = max(y_maxs)
        total_range = global_max - global_min if global_max > global_min else 1.0
        
        n = len(results_list)
        # 启发式偏移量：随数量增加而减小单步偏移
        offset_step = (total_range * 0.5) / (1 + 0.05 * n)
        
        for i, data in enumerate(plot_data):
            current_offset = i * offset_step
            line, = ax1.plot(data["V"], data["y"] + current_offset, label=data["name"], linewidth=1)
            color = line.get_color()
            
            # 绘制 Ev/Ec 标记
            if data["Ev"] is not None and not np.isnan(data["Ev"]):
                ax1.plot([data["Ev"]], [np.interp(data["Ev"], data["V"], data["y"]) + current_offset], 
                         marker='v', color=color, markersize=4)
            if data["Ec"] is not None and not np.isnan(data["Ec"]):
                ax1.plot([data["Ec"]], [np.interp(data["Ec"], data["V"], data["y"]) + current_offset], 
                         marker='^', color=color, markersize=4)

        ax1.set_ylabel(ylabel)
        ax1.set_title(t["overview"] + f" ({n} curves)")
        if self.show_annotations.get():
            # 优化多曲线模式下的图例显示
            ncol = 1
            if n > 15: ncol = 2
            if n > 30: ncol = 3
            leg_fontsize = 7
            if n > 20: leg_fontsize = 6
            if n > 40: leg_fontsize = 5
            
            ax1.legend(fontsize=leg_fontsize, loc='upper right', ncol=ncol, framealpha=0.5)
        
        try:
            self.fig1.tight_layout()
        except:
            pass
        self.canvas1.draw()

    def calculate_and_plot_average(self, results_list):
        """计算多个结果的平均值并重新分析显示"""
        if not results_list: return
        
        # 1. 确定基准 V 轴 (取第一个文件的 V 轴)
        base_v = results_list[0]["V"]
        didv_list = []
        i_list = []
        
        for res in results_list:
            v = res["V"]
            # 还原原始 dI/dV
            didv = 10**res["logS_raw"] - res["offset"]
            
            # 插值到基准 V 轴
            didv_interp = np.interp(base_v, v, didv)
            didv_list.append(didv_interp)
            
            if res["raw_I"] is not None:
                i_interp = np.interp(base_v, v, res["raw_I"])
                i_list.append(i_interp)
        
        # 2. 计算平均值
        avg_didv = np.mean(didv_list, axis=0)
        avg_i = np.mean(i_list, axis=0) if i_list else None
        
        # 3. 构造新的 DataFrame
        b_col = self.bias_col_var.get()
        c_col = self.curr_col_var.get()
        l_col = self.lix_col_var.get()
        
        data = {b_col: base_v, l_col: avg_didv}
        if avg_i is not None:
            data[c_col] = avg_i
        
        df_avg = pd.DataFrame(data)
        
        # 4. 调用对应的分析方法
        method_key = self.get_method_key()
        common_params = {
            "df": df_avg,
            "vcol": b_col,
            "icol": c_col if avg_i is not None else None,
            "lcol": l_col,
            "smooth_window": int(self.smooth_window.get()),
            "min_gap": float(self.min_gap.get())
        }
        
        if method_key == "Adaptive Thresholding":
            res_avg = detect_band_edges_dynamic(
                **common_params, 
                sigma_level=float(self.sigma_level.get()), 
                slope_percentile=float(self.slope_percentile.get()),
                v_center_min=float(self.v_center_min.get()),
                v_center_max=float(self.v_center_max.get()),
                delta_E=float(self.delta_E.get())
            )
        elif method_key == "Linear Extrapolation":
            res_avg = extract_gap_log_plateau(
                **common_params, 
                v_center_min=float(self.v_center_min.get()), 
                v_center_max=float(self.v_center_max.get()), 
                delta_E=float(self.delta_E.get()), 
                sigma_level=float(self.sigma_level.get())
            )
        elif method_key == "Slope Detection":
            res_avg = detect_band_edges_slope(
                **common_params,
                slope_threshold=float(self.slope_threshold.get()),
                v_center_min=float(self.v_center_min.get()),
                v_center_max=float(self.v_center_max.get()),
                fit_window=float(self.delta_E.get())
            )
        
        res_avg["filename"] = f"Averaged ({len(results_list)} files)"
        
        # 5. 绘图与显示
        self.plot_result(res_avg)
        self.show_info(res_avg)
        self.last_averaged_data = res_avg

    def copy_to_clipboard(self):
        """将当前分析的数据（单选或平均后的数据）复制到剪贴板"""
        lang = self.lang_var.get()
        t = TRANSLATIONS[lang]
        
        # 优先处理多选情况 (Waterfall 模式)
        if hasattr(self, 'current_results') and len(self.current_results) > 1 and not self.average_mode.get():
            header = "FileName\tEg(eV)\tEv(V)\tEc(V)\n"
            lines = [header]
            for res in self.current_results:
                row = f"{res['filename']}\t{res['Eg']:.6f}\t{res['Ev']:.6f}\t{res['Ec']:.6f}"
                lines.append(row)
            
            data_str = "\n".join(lines)
            self.master.clipboard_clear()
            self.master.clipboard_append(data_str)
            messagebox.showinfo(t["msg_finish"], t["msg_copied"])
            return

        res = self.last_averaged_data
        
        if res is None:
            messagebox.showwarning(t["msg_warn"], "No data available to copy.")
            return

        try:
            V = res["V"]
            # 还原 dI/dV (减去 offset)
            L = 10**res["logS_raw"] - res["offset"]
            I = res.get("raw_I")
            
            header = "Bias(V)\tdI/dV"
            if I is not None: header += "\tCurrent(A)"
            
            lines = [header]
            for i in range(len(V)):
                row = f"{V[i]:.6f}\t{L[i]:.6e}"
                if I is not None: row += f"\t{I[i]:.6e}"
                lines.append(row)
            
            data_str = "\n".join(lines)
            self.master.clipboard_clear()
            self.master.clipboard_append(data_str)
            messagebox.showinfo(t["msg_finish"], t["msg_copied"])
        except Exception as e:
            messagebox.showerror(t["msg_error"], f"Copy failed: {str(e)}")

    def show_multi_info(self, results):
        """在文本框显示多条曲线的摘要结果"""
        lang = self.lang_var.get()
        t = TRANSLATIONS[lang]
        
        self.info_text.delete("1.0", tk.END)
        header = f"{'FileName':<30}\t{'Eg(eV)':<8}\t{'Ev(V)':<8}\t{'Ec(V)':<8}\n"
        header += "-" * 65 + "\n"
        
        lines = [header]
        for res in results:
            name = res.get('filename', 'N/A')
            # 截断过长的文件名以保持对齐
            if len(name) > 28: name = name[:25] + "..."
            line = f"{name:<30}\t{res['Eg']:<8.4f}\t{res['Ev']:<8.4f}\t{res['Ec']:<8.4f}\n"
            lines.append(line)
        
        self.info_text.insert(tk.END, "".join(lines))

    def show_info(self, res):
        lang = self.lang_var.get()
        t = TRANSLATIONS[lang]
        
        self.info_text.delete("1.0", tk.END)
        display_method = t["methods"].get(res['method'], res['method'])
        info = f"Method: {display_method}\n"
        info += f"File: {res.get('filename', 'N/A')}\n"
        info += f"Eg: {res['Eg']:.4f} eV | Ev: {res['Ev']:.4f} V | Ec: {res['Ec']:.4f} V\n"
        if "floor_log" in res: 
            sigma_val = res['sigma']
            info += f"Base(log): {res['floor_log']:.4f} | Sigma(lin): {sigma_val:.3e}\n"
        if "threshold" in res: info += f"Threshold: {res['threshold']:.4f}\n"
        if "offset" in res: info += f"Vertical Offset: {res['offset']:.3e}\n"
        if "R2_vb" in res and res["R2_vb"] is not None: info += f"R2 VB: {res['R2_vb']:.4f} | R2 CB: {res['R2_cb']:.4f}\n"
        if res.get('method') == 'Slope Detection':
            r2vb = res.get('R2_vb', None)
            r2cb = res.get('R2_cb', None)
            info += f"R2 VB: {(r2vb if r2vb is not None else float('nan')):.4f} | R2 CB: {(r2cb if r2cb is not None else float('nan')):.4f}\n"
            info += f"Ev cand: {res.get('Ev_candidate', float('nan')):.4f} | Ec cand: {res.get('Ec_candidate', float('nan')):.4f}\n"
            info += f"Ev(curv): {res.get('Ev_curv', float('nan')):.4f} | curv_thr: {res.get('curv_thr', float('nan')):.3e}\n"
            info += f"Trigger Ev: {res.get('Ev_trigger', float('nan')):.4f} | Trigger Ec: {res.get('Ec_trigger', float('nan')):.4f}\n"
            info += f"Baseline: {res.get('y_floor', float('nan')):.3e} | Amp thr: {res.get('amp_thr', float('nan')):.3e}\n"

        info += f"Note: {res['note']}"
        self.info_text.insert(tk.END, info)

    def on_pick(self, event):
        if event.artist in [self.gap_min_line, self.gap_max_line]:
            self.dragging = True
            self.current_line = event.artist

    def on_motion(self, event):
        if self.dragging and event.xdata is not None:
            self.current_line.set_xdata([event.xdata, event.xdata])
            self.canvas2.draw()

    def on_release(self, event):
        if self.dragging:
            self.dragging = False
            self.v_center_min.set(f"{self.gap_min_line.get_xdata()[0]:.3f}")
            self.v_center_max.set(f"{self.gap_max_line.get_xdata()[0]:.3f}")
            self.current_line = None
            self.run_analysis()

if __name__ == "__main__":
    root = tk.Tk()
    app = LogGapGUI(root)
    root.mainloop()