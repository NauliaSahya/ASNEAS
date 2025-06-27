import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
st.set_page_config(
    page_title="CWT Viewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header Aplikasi ---
st.title("🌊 Continuous Wavelet Transform (CWT) Viewer")
st.subheader("Visualisasi sinyal, DFT, dan CWT")

# --- Pilih Sumber Sinyal ---
sig_src = st.radio(
    "🛠️ Pilih sumber sinyal:",
    ["Unggah file .txt", "Gunakan sinyal sinus", "Gunakan sinyal suara jantung sintetis"]
)

# --- Bagian Unggah File (jika dipilih) ---
up_file = None
if sig_src == "Unggah file .txt":
    up_file = st.file_uploader("📂 Unggah file sinyal (.txt)", type="txt")
    st.info("💡 Pastikan file TXT Anda berisi dua kolom: Waktu dan Amplitudo, dipisahkan oleh tab.")


# --- Parameter CWT di Sidebar ---
st.sidebar.header("⚙️ Parameter CWT")
f0 = st.sidebar.number_input("Frekuensi tengah (f0)", value=0.849, step=0.001, format="%.3f")
s0 = st.sidebar.number_input("Skala awal (s0)", value=0.014, step=0.0001, format="%.5f")
ds = st.sidebar.number_input("Delta skala (ds)", value=0.0004, step=0.00001, format="%.5f")
rowcount = st.sidebar.number_input("Jumlah skala (rowcount)", min_value=10, max_value=200, value=100, step=10)

# --- Sinyal dari sinus, file, atau sintetis ---
use_sig = False
fs = 1000 # Frekuensi sampling default
dt = 1 / fs # Langkah waktu default

if sig_src == "Gunakan sinyal sinus":
    st.sidebar.header("🎚️ Parameter Sinyal Sinus")
    sin_freq = st.sidebar.number_input("Frekuensi sinus (Hz)", value=50.0, min_value=0.1, max_value=1000.0, step=0.1)
    sin_amp = st.sidebar.number_input("Amplitudo sinus", value=2.0)
    sin_dur = st.sidebar.number_input("Durasi sinyal (s)", value=0.5, min_value=0.1, step=0.1)

    # Bangkitkan sinyal sinus
    time = np.arange(0, sin_dur, dt)
    x_t = sin_amp * np.sin(2 * np.pi * sin_freq * time)

    use_sig = True

elif sig_src == "Gunakan sinyal suara jantung sintetis":
    st.sidebar.header("❤️ Parameter Sinyal Jantung Sintetis")
    hr_bpm = st.sidebar.number_input("Detak Jantung (bpm)", value=70, min_value=30, max_value=180, step=1)
    sig_dur = st.sidebar.number_input("Durasi Sinyal (s)", value=3.0, min_value=1.0, max_value=10.0, step=0.5)

    # Bangkitkan sinyal suara jantung sintetis (model sederhana)
    # S1 biasanya lebih keras dan frekuensinya lebih rendah daripada S2
    # S2 mengikuti S1 setelah sistol, lalu diastol yang lebih panjang sebelum S1 berikutnya
    
    # Hitung interval detak
    beat_int_s = 60 / hr_bpm
    
    # Perkiraan waktu dalam satu detak (relatif terhadap S1)
    s1_dur = 0.08 # s
    s2_dur = 0.06 # s
    syst_dur_fact = 0.35 # Durasi sistol sebagai fraksi dari interval detak (misal, 35-40%)
    
    time = np.arange(0, sig_dur, dt) # Pastikan array waktu didefinisikan di sini
    x_t = np.zeros_like(time)
    
    curr_t = 0.0
    while curr_t < sig_dur:
        # S1 (bunyi "klik" frekuensi rendah)
        s1_start = curr_t
        s1_end = s1_start + s1_dur
        t_s1 = time[(time >= s1_start) & (time < s1_end)]
        if len(t_s1) > 0:
            x_t[(time >= s1_start) & (time < s1_end)] += 1.5 * np.exp(-1000 * (t_s1 - s1_start - s1_dur/2)**2) * np.sin(2 * np.pi * 30 * t_s1)
        
        # S2 (bunyi "klik" frekuensi lebih tinggi)
        s2_start = s1_start + beat_int_s * syst_dur_fact # S2 setelah sistol
        s2_end = s2_start + s2_dur
        t_s2 = time[(time >= s2_start) & (time < s2_end)]
        if len(t_s2) > 0:
            x_t[(time >= s2_start) & (time < s2_end)] += 1.0 * np.exp(-1000 * (t_s2 - s2_start - s2_dur/2)**2) * np.sin(2 * np.pi * 60 * t_s2)

        curr_t += beat_int_s

    # Tambahkan sedikit noise latar belakang
    x_t += 0.1 * np.random.randn(len(time))
    x_t = x_t / np.max(np.abs(x_t)) * 1.5 # Normalisasi amplitudo

    use_sig = True

elif up_file is not None:
    data = np.loadtxt(up_file, delimiter='\t')
    if data.shape[1] < 2:
        st.error("❌ File TXT harus memiliki setidaknya dua kolom: Waktu dan Amplitudo.")
        use_sig = False
    else:
        # Asumsi kolom pertama adalah waktu, kolom kedua adalah amplitudo sinyal
        time = data[:, 0]
        x_t = data[:, 1]
        
        # Hitung ulang fs dan dt berdasarkan array waktu data yang diunggah
        if len(time) > 1:
            dt = np.mean(np.diff(time))
            fs = 1 / dt
        else:
            st.warning("Sinyal terlalu pendek untuk menghitung frekuensi sampling yang akurat. Menggunakan default fs=1000.")
            fs = 1000
            dt = 1/fs

        n_time = len(x_t)
        # Pastikan array waktu konsisten dengan dt dan n_time jika tidak disediakan secara eksplisit atau tidak seragam
        time = np.linspace(0, n_time * dt, n_time)
        use_sig = True

# --- Jika sinyal tersedia ---
if use_sig:

    # Plot sinyal input (full range)
    st.subheader("📈 Sinyal Input Penuh")
    col_input = st.columns([1, 2, 1])[1]
    with col_input:
        fig_input = go.Figure()
        fig_input.add_trace(go.Scatter(x=time, y=x_t, mode='lines', name='Sinyal Input Penuh'))
        fig_input.update_layout(title='Sinyal Input Penuh', xaxis_title='Waktu (s)', yaxis_title='Amplitudo')
        st.plotly_chart(fig_input, use_container_width=True)

    # --- Slider untuk Rentang Sinyal yang Diproses CWT ---
    st.sidebar.header("⏱️ Rentang Waktu CWT")
    min_time_full = time[0]
    max_time_full = time[-1]
    
    if min_time_full >= max_time_full:
        st.warning("Durasi sinyal terlalu pendek untuk memilih rentang waktu.")
        selected_time_range = (min_time_full, max_time_full)
    else:
        selected_time_range = st.sidebar.slider(
            "Pilih rentang waktu untuk CWT (s):",
            float(min_time_full), float(max_time_full), 
            (float(min_time_full), float(max_time_full)) # Default to full range
        )
    
    # Filter sinyal berdasarkan selected_time_range
    start_idx_processed = np.searchsorted(time, selected_time_range[0])
    end_idx_processed = np.searchsorted(time, selected_time_range[1], side='right')
    
    processed_time = time[start_idx_processed:end_idx_processed]
    processed_x_t = x_t[start_idx_processed:end_idx_processed]
    
    if len(processed_time) == 0:
        st.warning("Rentang waktu yang dipilih tidak mengandung data sinyal. Sesuaikan slider atau pilih sumber sinyal yang berbeda.")
    else:
        st.info(f"Memproses CWT untuk rentang waktu: {selected_time_range[0]:.3f}s hingga {selected_time_range[1]:.3f}s")

        # Plot sinyal yang sedang diproses
        st.subheader("📊 Sinyal yang Diproses (Rentang Pilihan)")
        col_processed = st.columns([1, 2, 1])[1]
        with col_processed:
            fig_processed = go.Figure()
            fig_processed.add_trace(go.Scatter(x=processed_time, y=processed_x_t, mode='lines', name='Sinyal Diproses'))
            fig_processed.update_layout(title='Sinyal yang Dipilih untuk CWT', xaxis_title='Waktu (s)', yaxis_title='Amplitudo')
            st.plotly_chart(fig_processed, use_container_width=True)


        # --- DFT Manual (menggunakan sinyal yang diproses) ---
        st.subheader("🔊 Spektrum Frekuensi (DFT dari Sinyal Diproses)")

        def dft_manual(x, fs):
            N = len(x)
            freqs = np.arange(N) * fs / N
            dft_magnitude = np.zeros(N)
            for k in range(N):
                real = 0
                imag = 0
                for n in range(N):
                    angle = 2 * np.pi * k * n / N
                    real += x[n] * np.cos(-angle)
                    imag += x[n] * np.sin(-angle)
                dft_magnitude[k] = np.sqrt(real**2 + imag**2) /N
            return freqs[:N//2], dft_magnitude[:N//2]

        freqs_man, fft_vals_man = dft_manual(processed_x_t, fs)

        col_dft = st.columns([1, 2, 1])[1]
        with col_dft:
            fig_dft = go.Figure()
            fig_dft.add_trace(go.Scatter(x=freqs_man, y=fft_vals_man, mode='lines', name='Magnitudo'))
            fig_dft.update_layout(title='DFT Manual (Discrete Fourier Transform)',
                                  xaxis_title='Frekuensi (Hz)', yaxis_title='Magnitudo')
            st.plotly_chart(fig_dft, use_container_width=True)

        # Skala dan indeks
        s_idx = np.arange(1, rowcount + 1)
        scale = s0 + (s_idx * ds)
        w0 = 2 * np.pi * f0

        # Fungsi CWT yang baru
        @st.cache_data(show_spinner=True)
        def cwt_morlet(x, scale, time, w0): # Parameter dt_original tidak ada dan tidak digunakan
            cwt_re = np.zeros((len(time), len(scale)))
            cwt_im = np.zeros((len(time), len(scale)))
            for i, s in enumerate(scale):
                for j, τ in enumerate(time):
                    t_norm = (time - τ) / s
                    psi_re = (1 / np.sqrt(s)) * (np.pi**(-0.25)) * np.exp(-t_norm**2 / 2) * np.cos(w0 * t_norm)
                    psi_im = -(1 / np.sqrt(s)) * (np.pi**(-0.25)) * np.exp(-t_norm**2 / 2) * np.sin(w0 * t_norm)
                    cwt_re[j, i] = np.sum(x * psi_re) # <--- TIDAK ada perkalian dengan dt_original di sini
                    cwt_im[j, i] = np.sum(x * psi_im) # <--- TIDAK ada perkalian dengan dt_original di sini
            return np.sqrt(cwt_re**2 + cwt_im**2)

        st.info("⏳ Menghitung CWT...")
        cwt_mat = cwt_morlet(processed_x_t, scale, processed_time, w0) # Pass dt to the function

        # Plot Heatmap
        st.subheader("🌈 CWT 2D (Heatmap Skala vs Waktu)")
        col_hm = st.columns([1, 4, 1])[1]
        with col_hm:
            fig_hm = go.Figure(data=go.Heatmap(
                z=cwt_mat.T, # Transpose untuk orientasi yang benar (skala di y, waktu di x)
                x=processed_time,
                y=s_idx, # Menggunakan indeks skala untuk sumbu Y heatmap
                colorscale='Jet',
                colorbar_title='Magnitudo CWT'
            ))
            fig_hm.update_layout(
                height=1100, 
                xaxis_title='Waktu (s)',
                yaxis_title='Indeks Skala',
                title='CWT Heatmap'
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        # Plot Surface 3D
        st.subheader("🧊 CWT 3D Surface Plot")
        col_surf = st.columns([1, 4, 1])[1]
        with col_surf:
            T_surf, S_surf = np.meshgrid(processed_time, s_idx)
            Z_surf = cwt_mat.T # Transpose untuk orientasi sumbu Z yang benar

            fig_surf = go.Figure(data=[go.Surface(
                z=Z_surf,
                x=T_surf,
                y=S_surf,
                colorscale='Jet',
                colorbar_title='Magnitudo CWT'
            )])
            fig_surf.update_layout(
                height=1100, 
                scene=dict(
                    xaxis_title='Waktu (s)',
                    yaxis_title='Indeks Skala',
                    zaxis_title='Magnitudo CWT',
                    aspectratio=dict(x=1, y=1, z=0.5) 
                ),
                title='CWT Surface 3D'
            )
            st.plotly_chart(fig_surf, use_container_width=True)

        # Tabel Konversi Skala
        # Hubungan antara skala (s) dan pseudo-frekuensi (f_k) untuk wavelet Morlet adalah f_k = f0 / s
        fk = f0 / scale
        tk = (time[-1] / rowcount) * s_idx
        
        st.subheader("📊 Tabel Konversi Skala")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Indeks Skala → Frekuensi(Hz)**")
            df_freq = pd.DataFrame({
                "Indeks Skala": s_idx,
                "Frekuensi (Hz)": fk
            })
            st.dataframe(df_freq, use_container_width=True)

        with col2:
            st.markdown("**Indeks Skala → Periode(s)**")
            df_time = pd.DataFrame({
                "Indeks Skala": s_idx,
                "Waktu (s)": tk
            })
            st.dataframe(df_time, use_container_width=True)

        with st.expander("🔍 Lihat ringkasan matriks CWT"):
            st.write(f"Ukuran hasil CWT: {cwt_mat.shape} (Waktu x Skala)")
            st.dataframe(pd.DataFrame(cwt_mat), use_container_width=True)

        # --- Bagian Baru untuk Analisis Sinyal Suara Jantung ---
        st.header("❤️ Analisis Sinyal Suara Jantung")
        
        # --- (f). Envelope Curve dan Durasi ---
        st.subheader("Curve Envelope Sinyal & Durasi Waktu")

        st.info("💡 Curve Envelope membantu mengidentifikasi puncak sinyal seperti S1 dan S2.")

        # Filter Moving Average Manual untuk Envelope (menggantikan butter dan filtfilt)
        def manual_ma_filter(data, win_size):
            if win_size % 2 == 0:
                win_size += 1
            
            filt_data = np.zeros_like(data, dtype=float)
            half_win = win_size // 2
            
            for i in range(len(data)):
                start_idx = max(0, i - half_win)
                end_idx = min(len(data), i + half_win + 1)
                filt_data[i] = np.mean(data[start_idx:end_idx])
            return filt_data

        env_co_f = st.slider("Frekuensi Cutoff Filter Envelope (Hz)", 0.1, fs/2, value=45.0, step=1.0)
        env_win_size = int(fs / (2 * env_co_f))
        if env_win_size < 1: env_win_size = 1 

        # Terapkan filter moving average forward
        env_fwd = manual_ma_filter(np.abs(processed_x_t), env_win_size)
        # Terapkan filter moving average backward
        envelope = manual_ma_filter(env_fwd[::-1], env_win_size)[::-1]

        # Normalisasi envelope untuk visualisasi dan deteksi puncak yang lebih baik
        envelope = envelope / np.max(envelope) if np.max(envelope) > 0 else envelope

        # Plot sinyal dengan envelope
        fig_env = go.Figure()
        fig_env.add_trace(go.Scatter(x=processed_time, y=processed_x_t, mode='lines', name='Sinyal Asli', opacity=0.7))
        fig_env.add_trace(go.Scatter(x=processed_time, y=envelope, mode='lines', name='Curve Envelope', line=dict(color='red', width=2)))
        fig_env.update_layout(title='Sinyal Asli dan Curve Envelope', xaxis_title='Waktu (s)', yaxis_title='Amplitudo')

        # Deteksi Puncak Manual 
        def manual_find_peaks(sig, h_min=None, dist_min_s=0.1, samp_f=fs):
            pks = []
            # Konversi jarak dalam detik ke sampel
            dist_min_samp = max(1, int(dist_min_s * samp_f))
            
            last_pk_idx = -dist_min_samp 
            
            for i in range(1, len(sig) - 1):
                is_peak = (sig[i] > sig[i-1]) and (sig[i] > sig[i+1])
                
                if h_min is not None:
                    is_peak = is_peak and (sig[i] >= h_min)
                
                if is_peak:
                    # Terapkan batasan jarak
                    if (i - last_pk_idx) >= dist_min_samp:
                        pks.append(i)
                        last_pk_idx = i
                    elif pks and sig[i] > sig[pks[-1]]: # Jika lebih dekat tapi lebih tinggi, ganti
                        pks[-1] = i
                        last_pk_idx = i # Perbarui last_pk_idx untuk puncak baru yang lebih tinggi
            return np.array(pks)

        pk_h = st.slider("Tinggi Puncak Minimum (untuk S1/S2)", 0.01, 1.0, value=0.2, step=0.01)
        pk_dist_s = st.slider("Jarak Puncak Minimum (s)", 0.1, 2.0, value=0.3, step=0.05) 
        
        pks = manual_find_peaks(envelope, h_min=pk_h, dist_min_s=pk_dist_s, samp_f=fs)

        s1_pks = []
        s2_pks = []
        
        # Mulai dengan puncak kuat pertama sebagai S1. Ini mungkin memerlukan penyempurnaan untuk data nyata.
        if len(pks) >= 2:
            s1_pks.append(pks[0])
            for i in range(1, len(pks)):
                if len(s1_pks) > len(s2_pks): # Jika sebelumnya S1, berikutnya S2
                    s2_pks.append(pks[i])
                else: # Jika sebelumnya S2, berikutnya S1
                    s1_pks.append(pks[i])

        # Pastikan s1_pks dan s2_pks memiliki pasangan yang sesuai untuk perhitungan durasi
        min_len = min(len(s1_pks), len(s2_pks))
        s1_pks = s1_pks[:min_len]
        s2_pks = s2_pks[:min_len]
        
        # Plot penanda S1 dan S2
        if len(s1_pks) > 0: 
            fig_env.add_trace(go.Scatter(x=processed_time[s1_pks], y=envelope[s1_pks], mode='markers',
                                            marker=dict(symbol='circle', size=10, color='blue', line=dict(width=1, color='DarkSlateGrey')),
                                            name='Puncak S1'))
        if len(s2_pks) > 0: 
            fig_env.add_trace(go.Scatter(x=processed_time[s2_pks], y=envelope[s2_pks], mode='markers',
                                            marker=dict(symbol='star', size=10, color='green', line=dict(width=1, color='DarkSlateGrey')),
                                            name='Puncak S2'))

        st.plotly_chart(fig_env, use_container_width=True)

        st.markdown("**Durasi Waktu Sinyal Jantung**")
        durs_df_data = []

        # Hitung durasi
        for i in range(len(s1_pks)):
            s1_t = processed_time[s1_pks[i]]
            
            # Temukan S2 yang sesuai (harus setelah S1 dan sebelum S1 berikutnya jika tersedia)
            s2_t = np.nan
            syst_dur = np.nan
            diast_dur = np.nan
            sgl_cyc_dur = np.nan

            s2_found = False
            for j in range(len(s2_pks)):
                if processed_time[s2_pks[j]] > s1_t: # Cari S2 setelah S1 saat ini
                    s2_t = processed_time[s2_pks[j]]
                    syst_dur = s2_t - s1_t
                    s2_found = True
                    break
            
            if s2_found:
                # Cari S1 berikutnya setelah S2 yang ditemukan, atau gunakan akhir sinyal yang diproses
                next_s1_t_to_use = np.nan
                next_s1_found = False
                for k in range(i + 1, len(s1_pks)):
                    if processed_time[s1_pks[k]] > s2_t: # Cari S1 berikutnya setelah S2
                        next_s1_t_to_use = processed_time[s1_pks[k]]
                        next_s1_found = True
                        break
                
                if next_s1_found:
                    sgl_cyc_dur = next_s1_t_to_use - s1_t
                    diast_dur = next_s1_t_to_use - s2_t
                else:
                    # Jika tidak ada S1 berikutnya dalam rentang yang diproses, gunakan akhir sinyal
                    # Ini mungkin tidak merepresentasikan siklus penuh yang sebenarnya, tapi adalah perkiraan terbaik
                    sgl_cyc_dur = processed_time[-1] - s1_t
                    diast_dur = processed_time[-1] - s2_t

            # Format output
            s2_t_str = f"{s2_t:.3f}" if not np.isnan(s2_t) else "N/A"
            syst_dur_str = f"{syst_dur:.3f}" if not np.isnan(syst_dur) else "N/A"
            diast_dur_str = f"{diast_dur:.3f}" if not np.isnan(diast_dur) else "N/A"
            sgl_cyc_dur_str = f"{sgl_cyc_dur:.3f}" if not np.isnan(sgl_cyc_dur) else "N/A"

            durs_df_data.append({
                "Siklus": i + 1,
                "Waktu S1 (s)": f"{s1_t:.3f}",
                "Waktu S2 (s)": s2_t_str,
                "Durasi Sistolik (s)": syst_dur_str,
                "Durasi Diastolik (s)": diast_dur_str,
                "Durasi Satu Siklus (s)": sgl_cyc_dur_str
            })


        durs_df = pd.DataFrame(durs_df_data)
        st.dataframe(durs_df, use_container_width=True)
        
        # --- (g). Thresholding CWT Kontur & Identifikasi Komponen ---
        st.subheader("Thresholding Kontur CWT & Identifikasi Komponen Jantung")

        st.info("💡 Thresholding membantu memfokuskan pada area energi tinggi. Identifikasi komponen A2, P2, M1, T1 dilakukan berdasarkan posisi relatif terhadap S1/S2 dan frekuensi.")

        cwt_max = np.max(cwt_mat)
        cwt_thresh_pct = st.slider("Threshold CWT (dalam % dari Maks)", 0, 100, value=30, step=5)
        cwt_thresh = cwt_max * (cwt_thresh_pct / 100.0)

        # Slider baru untuk ambang batas relatif identifikasi komponen
        min_magnitude_ratio = st.slider("Rasio Magnitudo Minimum untuk Komponen (dari maks di jendela)", 0.00, 1.00, value=0.99, step=0.01)
        st.info(f"💡 Hanya titik CWT yang magnitudonya lebih besar dari {min_magnitude_ratio*100:.1f}% dari magnitudo maksimum di window waktu-frekuensi lokal yang akan dianggap sebagai komponen.")


        # Buat salinan dari figur heatmap untuk menambahkan titik-titik yang di-threshold dan komponen
        fig_comps = go.Figure(data=go.Heatmap(
            z=cwt_mat.T,
            x=processed_time, 
            y=s_idx,
            colorscale='Jet',
            colorbar_title='Magnitudo CWT'
        ))
        fig_comps.update_layout(
            height=1100, 
            xaxis_title='Waktu (s)',
            yaxis_title='Indeks Skala',
            title='CWT Heatmap dengan Kontur dan Komponen Jantung',
            # Tambahkan konfigurasi legenda untuk memindahkannya ke bagian bawah
            legend=dict(
                orientation="h",  
                yanchor="bottom", 
                y=-0.3,           
                xanchor="left",   
                x=0               
            )
        )

        # Plot titik-titik yang di-threshold
        thresh_idx = np.argwhere(cwt_mat > cwt_thresh)
        if len(thresh_idx) > 0:
            thresh_t = processed_time[thresh_idx[:, 0]]
            thresh_s_idx = s_idx[thresh_idx[:, 1]]
            
            fig_comps.add_trace(go.Scatter(x=thresh_t, y=thresh_s_idx, mode='markers',
                                            marker=dict(symbol='diamond-open', size=4, color='white', opacity=0.9),
                                            name=f'Kontur > {cwt_thresh_pct}% Maks'))
        
        # Identifikasi dan plot A2, P2, M1, T1 komponen
        
        comp_marks = []

        # Definisikan rentang frekuensi perkiraan untuk komponen (dalam Hz)
        # S1 components (M1, T1) have lower frequencies. M1 and T1 are normally audible components of S1.
        freq_m1_t1 = (10, 50) 
        # S2 components (A2, P2) have higher frequencies. A2 and P2 are components of S2.
        freq_a2_p2 = (25, 70) 

        # Konversi rentang frekuensi ke rentang indeks skala
        # skala = f0 / freq -> frekuensi lebih rendah berarti indeks skala lebih tinggi
        s_idx_m1_t1_min = np.where(fk <= freq_m1_t1[1])[0].min() if np.any(fk <= freq_m1_t1[1]) else 0
        s_idx_m1_t1_max = np.where(fk >= freq_m1_t1[0])[0].max() if np.any(fk >= freq_m1_t1[0]) else len(s_idx)-1

        s_idx_a2_p2_min = np.where(fk <= freq_a2_p2[1])[0].min() if np.any(fk <= freq_a2_p2[1]) else 0
        s_idx_a2_p2_max = np.where(fk >= freq_a2_p2[0])[0].max() if np.any(fk >= freq_a2_p2[0]) else len(s_idx)-1

        
        # Iterasi melalui puncak S1 dan S2 yang terdeteksi
        for i in range(len(s1_pks)):
            s1_t_idx = s1_pks[i]
            s1_t = processed_time[s1_t_idx] 
            
            s2_t_idx = -1
            # Temukan S2 yang sesuai (harus setelah S1 dan sebelum S1 berikutnya jika tersedia)
            for j in range(len(s2_pks)):
                if processed_time[s2_pks[j]] > s1_t: 
                    s2_t_idx = s2_pks[j]
                    s2_t = processed_time[s2_t_idx] 
                    break
            
            # --- Analisis wilayah S1 untuk M1 dan T1 ---
            # Definisikan jendela kecil di sekitar puncak S1 untuk pencarian komponen
            s1_win_center_time = s1_t
            s1_win_half_width = 0.08 
            
            s1_win_start_time = s1_win_center_time - s1_win_half_width
            s1_win_end_time = s1_win_center_time + s1_win_half_width

            s1_win_start_idx = np.searchsorted(processed_time, s1_win_start_time)
            s1_win_end_idx = np.searchsorted(processed_time, s1_win_end_time, side='right')
            
            if s1_win_end_idx > s1_win_start_idx:
                # Ekstrak data CWT untuk jendela S1 dan skala yang relevan
                s1_cwt_win = cwt_mat[s1_win_start_idx:s1_win_end_idx, s_idx_m1_t1_min : s_idx_m1_t1_max + 1]
                
                if s1_cwt_win.size > 0:
                    # Temukan puncak di wilayah ini, prioritaskan magnitudo tertinggi
                    win_max_mag_s1 = np.max(s1_cwt_win)
                    
                    if win_max_mag_s1 > 0:
                        # Temukan indeks dalam jendela yang berada di atas ambang batas relatif
                        # Gunakan min_magnitude_ratio baru untuk deteksi komponen
                        local_rel_thresh_s1 = min_magnitude_ratio * win_max_mag_s1 
                        
                        s1_comp_indices_in_win = np.argwhere(s1_cwt_win > local_rel_thresh_s1)
                        
                        if s1_comp_indices_in_win.shape[0] > 0:
                            # Konversi indeks lokal kembali ke indeks waktu dan skala global
                            global_t_indices_s1 = s1_win_start_idx + s1_comp_indices_in_win[:, 0]
                            global_s_indices_s1 = s_idx_m1_t1_min + s1_comp_indices_in_win[:, 1]
                            
                            m1_t_idx, m1_s_idx = None, None
                            t1_t_idx, t1_s_idx = None, None

                            # Urutkan berdasarkan waktu (menaik) untuk M1 dan T1, lalu berdasarkan magnitudo (menurun)
                            # Ini memprioritaskan kejadian yang lebih awal jika magnitudo serupa
                            sorted_order_s1 = np.lexsort((-cwt_mat[global_t_indices_s1, global_s_indices_s1], global_t_indices_s1))
                            sorted_global_t_indices_s1 = global_t_indices_s1[sorted_order_s1]
                            sorted_global_s_indices_s1 = global_s_indices_s1[sorted_order_s1]

                            # Cari M1 (titik kuat paling awal)
                            if len(sorted_global_t_indices_s1) > 0:
                                m1_t_idx = sorted_global_t_indices_s1[0]
                                m1_s_idx = sorted_global_s_indices_s1[0]
                                comp_marks.append({'t_idx': m1_t_idx, 's_idx': m1_s_idx, 'name': f'M1 (Siklus {i+1})', 'color': 'yellow', 'symbol': 'star'})
                            
                            # Cari T1 (titik kuat kedua paling awal, setelah M1)
                            if m1_t_idx is not None:
                                for k in range(1, len(sorted_global_t_indices_s1)):
                                    current_global_t_idx = sorted_global_t_indices_s1[k]
                                    current_global_s_idx = sorted_global_s_indices_s1[k]
                                    
                                    # Pastikan T1 berbeda dalam waktu dari M1 (misal, > 10ms pemisahan)
                                    if (processed_time[current_global_t_idx] - processed_time[m1_t_idx]) > 0.01:
                                        t1_t_idx = current_global_t_idx
                                        t1_s_idx = current_global_s_idx
                                        comp_marks.append({'t_idx': t1_t_idx, 's_idx': t1_s_idx, 'name': f'T1 (Siklus {i+1})', 'color': 'blue', 'symbol': 'star-open'})
                                        break # Ditemukan T1, keluar dari loop dalam untuk komponen S1
                                
            # --- Analisis wilayah S2 untuk A2 dan P2 ---
            if s2_t_idx != -1:
                s2_win_center_time = s2_t
                s2_win_half_width = 0.08 # s
                
                s2_win_start_time = s2_win_center_time - s2_win_half_width
                s2_win_end_time = s2_win_center_time + s2_win_half_width

                s2_win_start_idx = np.searchsorted(processed_time, s2_win_start_time)
                s2_win_end_idx = np.searchsorted(processed_time, s2_win_end_time, side='right')
                
                if s2_win_end_idx > s2_win_start_idx:
                    s2_cwt_win = cwt_mat[s2_win_start_idx:s2_win_end_idx, s_idx_a2_p2_min : s_idx_a2_p2_max + 1]
                    
                    if s2_cwt_win.size > 0:
                        win_max_mag_s2 = np.max(s2_cwt_win)
                        if win_max_mag_s2 > 0:
                            local_rel_thresh_s2 = min_magnitude_ratio * win_max_mag_s2
                            s2_comp_indices_in_win = np.argwhere(s2_cwt_win > local_rel_thresh_s2)
                            
                            if s2_comp_indices_in_win.shape[0] > 0:
                                global_t_indices_s2 = s2_win_start_idx + s2_comp_indices_in_win[:, 0]
                                global_s_indices_s2 = s_idx_a2_p2_min + s2_comp_indices_in_win[:, 1]
                                
                                a2_t_idx, a2_s_idx = None, None
                                p2_t_idx, p2_s_idx = None, None

                                # Urutkan berdasarkan waktu (menaik) untuk A2 dan P2, lalu berdasarkan magnitudo (menurun)
                                sorted_order_s2 = np.lexsort((-cwt_mat[global_t_indices_s2, global_s_indices_s2], global_t_indices_s2))
                                sorted_global_t_indices_s2 = global_t_indices_s2[sorted_order_s2]
                                sorted_global_s_indices_s2 = global_s_indices_s2[sorted_order_s2]

                                # Cari A2 (titik kuat paling awal)
                                if len(sorted_global_t_indices_s2) > 0:
                                    a2_t_idx = sorted_global_t_indices_s2[0]
                                    a2_s_idx = sorted_global_s_indices_s2[0]
                                    comp_marks.append({'t_idx': a2_t_idx, 's_idx': a2_s_idx, 'name': f'A2 (Siklus {i+1})', 'color': 'cyan', 'symbol': 'circle'})
                                
                                # Cari P2 (titik kuat kedua paling awal, setelah A2)
                                if a2_t_idx is not None:
                                    for k in range(1, len(sorted_global_t_indices_s2)):
                                        current_global_t_idx = sorted_global_t_indices_s2[k]
                                        current_global_s_idx = sorted_global_s_indices_s2[k]

                                        # Pastikan P2 berbeda dalam waktu dari A2 (misal, > 10ms pemisahan)
                                        if (processed_time[current_global_t_idx] - processed_time[a2_t_idx]) > 0.01:
                                            p2_t_idx = current_global_t_idx
                                            p2_s_idx = current_global_s_idx
                                            comp_marks.append({'t_idx': p2_t_idx, 's_idx': p2_s_idx, 'name': f'P2 (Siklus {i+1})', 'color': 'purple', 'symbol': 'circle-open'})
                                            break # Ditemukan P2, keluar dari loop dalam untuk komponen S2
                                


        # Tambahkan penanda komponen ke plot
        for comp in comp_marks:
            fig_comps.add_trace(go.Scatter(
                x=[processed_time[comp['t_idx']]],
                y=[s_idx[comp['s_idx']]],
                mode='markers',
                marker=dict(symbol=comp['symbol'], size=12, color=comp['color'], line=dict(width=2, color='black')),
                name=comp['name']
            ))

        st.plotly_chart(fig_comps, use_container_width=True)

        marker_data = []
        for comp in comp_marks:
            # Dapatkan nilai x (waktu), y (indeks skala), dan z (magnitudo CWT)
            time_val = processed_time[comp['t_idx']]
            scale_idx_val = s_idx[comp['s_idx']]
            cwt_magnitude_val = cwt_mat[comp['t_idx'], comp['s_idx']] # cwt_mat memiliki dimensi (waktu, skala)

            marker_data.append({
                "Nama Komponen": comp['name'],
                "Waktu (s)": f"{time_val:.3f}",
                "Indeks Skala": f"{scale_idx_val:.2f}",
                "Magnitudo CWT": f"{cwt_magnitude_val:.3f}"
            })

        if marker_data:
            df_markers = pd.DataFrame(marker_data)
            st.dataframe(df_markers, use_container_width=True)
        else:
            st.info("Tidak ada komponen M1, T1, A2, atau P2 yang teridentifikasi berdasarkan kriteria saat ini.")

        st.markdown("""
        **Catatan Penting untuk Identifikasi Komponen (M1, T1, A2, P2):**
        * Akurasi sangat bergantung pada kualitas sinyal, parameter CWT, dan deteksi puncak S1/S2.
        * **M1** (penutupan katup mitral) dan **T1** (penutupan katup trikuspid) membentuk **S1**. M1 umumnya sedikit mendahului T1.
        * **A2** (penutupan katup aorta) dan **P2** (penutupan katup pulmonal) membentuk **S2**. A2 umumnya sedikit mendahului P2. Pemisahan A2-P2 sering disebut 'pemisahan fisiologis' dan lebih terlihat saat inspirasi.
        """)

        