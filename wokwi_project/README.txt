Wokwi project: cosine_predictor_esp32

Files included:
- cosine_predictor.ino  (root sketch)
- weights.h            (float weight arrays used by the sketch)
- model.h              (full TFLite binary as C array; included for completeness)
- diagram.json         (Wokwi project manifest with parts and connections)

Notes and verification:
- All three files referenced in `diagram.json` exist in this folder.
- `cosine_predictor.ino` includes `weights.h` and prints CSV (`predicted,actual`) and JSON-line test results. It does not read sensors connected in the diagram (DHT22, LDR, PIR, HC-SR04, etc.). If you need those sensors to be read, add code to `cosine_predictor.ino` to initialize and poll them.
- `model.h` is not used by the current manual inference sketch but is included because the assignment asked to embed the converted TFLite model.

How to import into Wokwi:
1. Zip the contents of this folder (diagram.json + files).
2. In Wokwi: Import Project -> Upload ZIP.
3. Open the Serial Monitor / Serial Plotter at 115200 baud to view output.

Quick PowerShell to create ZIP from workspace root:

```powershell
Compress-Archive -Path .\wokwi_project\* -DestinationPath .\wokwi_project.zip -Force
```

If you want, I can create the ZIP here now. If you'd like the sensors wired into the sketch, tell me which sensors to support and I'll add example code.`