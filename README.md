# Purwarupa Deteksi Anomali Gas CO dan CO2

Repository ini memisahkan akuisisi, pemrosesan, model, dan dashboard. Sensor aktif adalah Winsen ZE07-CO melalui UART dan DFRobot Gravity STCC4 SEN0678 melalui I2C. SCD41 tidak digunakan oleh alur aktif.

## Arsitektur

- `src/sensors/`: adapter ZE07-CO dan STCC4. Adapter tidak membuka hardware ketika modul diimpor.
- `src/acquisition_service.py`: proses terpisah yang membaca DUMMY atau SENSOR setiap 1 detik, memvalidasi status, menjalankan SMA causal 5 sampel, inference Isolation Forest, dan menulis CSV. Proses ini tidak melatih model secara otomatis.
- `src/acquisition_service.py --mode collect-baseline`: proses akuisisi baseline sensor-only; tidak memuat, melatih, atau menjalankan Isolation Forest.
- `src/sensor_pipeline.py`: pipeline validasi -> SMA -> fitur `co_ma`, `co2_ma` -> inference -> warning.
- `src/offline_training.py`: review dan training offline dari CSV baseline normal yang dikumpulkan terpisah.
- `src/dashboard.py`: Streamlit read-only terhadap CSV; browser refresh tidak menginisialisasi sensor atau melatih model.
- `src/sensors/vendor/DFRobot_STCC4.py`: kode driver DFRobot resmi yang divendorkan dengan atribusi dan lisensi MIT.

Threshold 35 ppm CO dan 1000 ppm CO2 adalah parameter simulasi operasional, bukan klaim batas kesehatan sesaat. Threshold alert memakai nilai sensor valid sebelum SMA. Data invalid, termasuk status STCC4 256/0x0100 yang belum terverifikasi pada unit fisik, tidak masuk SMA atau model.

Dashboard menampilkan dua indikator berbeda: `if_status` (`NOT READY`, `NORMAL`, `ANOMALY`) adalah hasil Isolation Forest, sedangkan `safety_status` (`NORMAL`, `EARLY WARNING`, `THRESHOLD ALERT`, `SENSOR ERROR`) mengikuti kebijakan operasional. Anomali statistik tidak otomatis berarti kondisi berbahaya.

## Mode simulasi Windows

Install dependensi:

```powershell
python -m pip install -r requirements.txt
```

Jika sudah memiliki baseline normal sensor, tinjau lalu latih model secara offline:

```powershell
python -m src.offline_training baseline_sensor_normal.csv --model data/models/isolation_forest.joblib --review
python -m src.offline_training baseline_sensor_normal.csv --model data/models/isolation_forest.joblib --approve
```

Jalankan acquisition di terminal pertama:

```powershell
python -m src.acquisition_service --mode dummy --scenario normal
```

Jalankan dashboard di terminal kedua:

```powershell
python -m streamlit run src/dashboard.py --server.port 8501
```

Buka `http://localhost:8501`. Skenario dapat dipilih melalui kontrol dashboard; tombol Start/Pause/Reset mengirim perintah ke acquisition service melalui file kontrol. Acquisition service menolak berjalan jika model belum dilatih offline.

Smoke test singkat dengan model yang sudah tersedia:

```powershell
python -m src.acquisition_service --mode dummy --scenario sudden_spike --max-samples 8 --output data/smoke.csv --model data/models/isolation_forest.joblib
```

## Deployment Raspberry Pi 4

Aktifkan I2C, pastikan ZE07-CO terhubung ke UART `/dev/serial0` pada 9600 baud, dan verifikasi STCC4 pada bus 1/alamat `0x64` (`i2cdetect -y 1`). Install dependensi, review baseline, lalu jalankan acquisition sebagai proses terpisah:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.offline_training baseline_sensor_normal.csv --model data/models/isolation_forest.joblib --review
python3 -m src.offline_training baseline_sensor_normal.csv --model data/models/isolation_forest.joblib --approve
python3 -m src.acquisition_service --mode sensor
```

Untuk akses dashboard dari laptop/HP pada jaringan yang sama:

```bash
python3 -m streamlit run src/dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Kemudian buka `http://ALAMAT_IP_RASPBERRY_PI:8501`. Jangan menjalankan `--mode sensor` di Windows.

## Mengumpulkan baseline langsung dari sensor

Mode baseline tidak membutuhkan model yang sudah dilatih. Jalankan hanya pada Raspberry Pi setelah ZE07-CO dan STCC4 terhubung:

```bash
python3 -m src.acquisition_service \
  --mode collect-baseline \
  --duration 1800 \
  --output data/baseline_sensor_normal.csv
```

Sampling menggunakan jadwal monotonic nominal 1 detik. File dibuka dalam mode append dan header/schema yang tidak cocok akan ditolak untuk mencegah pencampuran data. Data yang tersimpan mencakup timestamp, CO, CO2, temperatur, kelembapan, status STCC4, validitas dan error masing-masing sensor (`co_valid`, `co_error`, `co2_valid`, `co2_error`), validitas gabungan, source, dan waktu monotonic. Retry terbatas digunakan untuk timeout transien; tidak ada nilai sintetis atau pengisian NaN. `Ctrl+C` menghentikan loop dengan aman dan mempertahankan baris yang sudah tertulis.

Setelah selesai, tinjau CSV tersebut sebelum training:

```bash
python3 -m src.offline_training \
  data/baseline_sensor_normal.csv \
  --model data/models/isolation_forest.joblib \
  --review
```

## Prosedur pengumpulan baseline normal

1. Jalankan sensor pada kondisi kabin yang dinyatakan normal dan stabil, tanpa kejadian gas yang sengaja dibuat.
2. Tunggu warm-up sensor dan catat sekurang-kurangnya beberapa puluh menit dengan interval target 1 detik. Dokumentasikan waktu, kondisi kabin, ventilasi, temperatur, kelembapan, dan status sensor.
3. Simpan CSV dengan kolom minimal `timestamp`, `co_ppm`, dan `co2_ppm`. Kolom `source=SENSOR` dan catatan kondisi pengukuran disarankan.
4. Tinjau statistik, timestamp duplikat, nilai invalid, status sensor, dan indikasi kejadian abnormal:

```powershell
python -m src.offline_training baseline_sensor_normal.csv --model data/models/isolation_forest.joblib --review
```

5. Hanya setelah dataset disetujui secara metodologis, jalankan `--approve`. Dataset testing, data threshold crossing, spike, dan data abnormal tidak boleh berada dalam baseline training.

## Pengujian

```powershell
python -m pytest -q -p no:cacheprovider
```

Training selalu menggunakan data normal terpisah. Suhu dan kelembapan disimpan ke CSV/dashboard, tetapi bukan fitur Isolation Forest. Driver STCC4 dipertahankan sesuai atribusi DFRobot dan MIT License.
