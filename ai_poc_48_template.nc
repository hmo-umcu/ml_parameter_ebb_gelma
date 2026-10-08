% ai_poc_48
; REGENHU
; http://www.regenhu.com/
; Translator 1.0.0.0
; Date 2026-04-13
; ESTIMATED_PRINTING_TIME 0:03:19.061356

; INITIALIZATION
T1
M200=800 ; Set pressure to 80kPa
T0
G803 ; Move to safe height
; INITIALIZATION

M110=0 ; Set printing progress to 0%
T1
G801 ; Measure tool

#CONTOUR MODE ON [DEV PATH_DEV=0.08]

M312 ; Wait for work zone temperature
G805[-45.785, 32.720, 2.620] ; Set G55 origin
G55

; Changing tool to 'PSD 1'
#FLUSH WAIT
T1
G807[1, 0.002, 0.002] ; Enable time-based start (0.002) / stop (0.002) delays [s]
M200=800 ; Set pressure to 80kPa
F10.000
M302 ; Wait for tool temperature
G00 G55 X-1.900 Y1.400
M151 ; Engage tool for printing
Z0.600
M110=40 ; Set printing progress to 4%
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
M110=50 ; Set printing progress to 5%
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-45.785, 19.640, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M110=60 ; Set printing progress to 6%
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
M110=70 ; Set printing progress to 7%
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-45.785, 6.560, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M110=80 ; Set printing progress to 8%
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
M110=90 ; Set printing progress to 9%
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-45.785, -6.520, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M110=100 ; Set printing progress to 10%
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
X1.900
M110=110 ; Set printing progress to 11%
Y-1.400
M161 ; Turn off dispensing
G805[-45.785, -19.600, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M110=120 ; Set printing progress to 12%
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M110=130 ; Set printing progress to 13%
M161 ; Turn off dispensing
G805[-45.785, -32.680, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=140 ; Set printing progress to 14%
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M110=150 ; Set printing progress to 15%
M161 ; Turn off dispensing
G805[-32.705, -32.680, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=160 ; Set printing progress to 16%
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M110=170 ; Set printing progress to 17%
M161 ; Turn off dispensing
G805[-32.705, -19.600, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=180 ; Set printing progress to 18%
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-32.705, -6.520, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=190 ; Set printing progress to 19%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=200 ; Set printing progress to 20%
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-32.705, 6.560, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=210 ; Set printing progress to 21%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=220 ; Set printing progress to 22%
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-32.705, 19.640, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=230 ; Set printing progress to 23%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=240 ; Set printing progress to 24%
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-32.705, 32.720, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=250 ; Set printing progress to 25%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=260 ; Set printing progress to 26%
X-1.000
Y1.900
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-19.625, 32.720, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=270 ; Set printing progress to 27%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
M110=280 ; Set printing progress to 28%
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-19.625, 19.640, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=290 ; Set printing progress to 29%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
M110=300 ; Set printing progress to 30%
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-19.625, 6.560, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=310 ; Set printing progress to 31%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
M110=320 ; Set printing progress to 32%
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-19.625, -6.520, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=330 ; Set printing progress to 33%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
M110=340 ; Set printing progress to 34%
X0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-19.625, -19.600, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=350 ; Set printing progress to 35%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=360 ; Set printing progress to 36%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-19.625, -32.680, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=370 ; Set printing progress to 37%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=380 ; Set printing progress to 38%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-6.545, -32.680, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=390 ; Set printing progress to 39%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=400 ; Set printing progress to 40%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-6.545, -19.600, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=410 ; Set printing progress to 41%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=420 ; Set printing progress to 42%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-6.545, -6.520, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=430 ; Set printing progress to 43%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=440 ; Set printing progress to 44%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-6.545, 6.560, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=450 ; Set printing progress to 45%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=460 ; Set printing progress to 46%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-6.545, 19.640, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=470 ; Set printing progress to 47%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=480 ; Set printing progress to 48%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[-6.545, 32.720, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=490 ; Set printing progress to 49%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=500 ; Set printing progress to 50%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[6.535, 32.720, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M110=510 ; Set printing progress to 51%
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
M110=520 ; Set printing progress to 52%
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[6.535, 19.640, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M110=530 ; Set printing progress to 53%
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
M110=540 ; Set printing progress to 54%
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[6.535, 6.560, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M110=550 ; Set printing progress to 55%
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
M110=560 ; Set printing progress to 56%
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[6.535, -6.520, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M110=570 ; Set printing progress to 57%
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
M110=580 ; Set printing progress to 58%
Y-1.400
M161 ; Turn off dispensing
G805[6.535, -19.600, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M110=590 ; Set printing progress to 59%
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M110=600 ; Set printing progress to 60%
M161 ; Turn off dispensing
G805[6.535, -32.680, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=610 ; Set printing progress to 61%
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M110=620 ; Set printing progress to 62%
M161 ; Turn off dispensing
G805[19.615, -32.680, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=630 ; Set printing progress to 63%
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M110=640 ; Set printing progress to 64%
M161 ; Turn off dispensing
G805[19.615, -19.600, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=650 ; Set printing progress to 65%
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M110=660 ; Set printing progress to 66%
M161 ; Turn off dispensing
G805[19.615, -6.520, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=670 ; Set printing progress to 67%
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M110=680 ; Set printing progress to 68%
M161 ; Turn off dispensing
G805[19.615, 6.560, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=690 ; Set printing progress to 69%
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[19.615, 19.640, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=700 ; Set printing progress to 70%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=710 ; Set printing progress to 71%
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[19.615, 32.720, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=720 ; Set printing progress to 72%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
M110=730 ; Set printing progress to 73%
X-1.000
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[32.695, 32.720, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=740 ; Set printing progress to 74%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
M110=750 ; Set printing progress to 75%
Y1.900
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[32.695, 19.640, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=760 ; Set printing progress to 76%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
M110=770 ; Set printing progress to 77%
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[32.695, 6.560, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=780 ; Set printing progress to 78%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
M110=790 ; Set printing progress to 79%
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[32.695, -6.520, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=800 ; Set printing progress to 80%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
M110=810 ; Set printing progress to 81%
X-0.000
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[32.695, -19.600, 2.620] ; Set G55 origin
G55
G00 Z18.400
M110=820 ; Set printing progress to 82%
X-1.900 Y1.400
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X-0.000
M110=830 ; Set printing progress to 83%
Y-1.900
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[32.695, -32.680, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=840 ; Set printing progress to 84%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X-0.000
Y-1.900
M110=850 ; Set printing progress to 85%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[45.775, -32.680, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=860 ; Set printing progress to 86%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=870 ; Set printing progress to 87%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[45.775, -19.600, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=880 ; Set printing progress to 88%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=890 ; Set printing progress to 89%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[45.775, -6.520, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=900 ; Set printing progress to 90%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=910 ; Set printing progress to 91%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[45.775, 6.560, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=920 ; Set printing progress to 92%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=930 ; Set printing progress to 93%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[45.775, 19.640, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=940 ; Set printing progress to 94%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=950 ; Set printing progress to 95%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing
G805[45.775, 32.720, 2.620] ; Set G55 origin
G55
G00 Z18.400
X-1.900 Y1.400
M110=960 ; Set printing progress to 96%
Z0.600
M160 ; Turn on dispensing
G01 Y-1.900
X-1.000
Y1.900
X0.000
Y-1.900
M110=970 ; Set printing progress to 97%
X1.000
Y1.900
X1.900
Y-1.400
M161 ; Turn off dispensing

#FLUSH WAIT

; ============================================================
; CAMERA IMAGING POSITIONS
; Wells: A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2, A3, B3, C3, D3, E3, F3, A4, B4, C4, D4, E4, F4, A5, B5, C5, D5, E5, F5, A6, B6, C6, D6, E6, F6, A7, B7, C7, D7, E7, F7, A8, B8, C8, D8, E8, F8
; Slot 1 selected (T1) to match coordinate frame of recorded
; positions. Camera physically mounted at slot 5.
; Z safe travel = 40.0 mm  |  Z imaging = 20.0 mm
; ============================================================

#FLUSH WAIT
#CONTOUR MODE OFF          ; Exit tracking mode from printing
#FLUSH WAIT

T1                         ; Select slot 1 (matches recorded coordinate frame)
G803                       ; Move to system safe height

; --- Well A1 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-250.160 Y53.250              ; Move XY to A1 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at A1 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well B1 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-250.160 Y40.242              ; Move XY to B1 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at B1 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well C1 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-250.160 Y27.234              ; Move XY to C1 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at C1 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well D1 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-250.160 Y14.226              ; Move XY to D1 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at D1 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well E1 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-250.160 Y1.218              ; Move XY to E1 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at E1 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well F1 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-250.160 Y-11.790              ; Move XY to F1 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at F1 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well A2 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-237.023 Y53.250              ; Move XY to A2 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at A2 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well B2 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-237.023 Y40.242              ; Move XY to B2 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at B2 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well C2 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-237.023 Y27.234              ; Move XY to C2 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at C2 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well D2 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-237.023 Y14.226              ; Move XY to D2 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at D2 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well E2 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-237.023 Y1.218              ; Move XY to E2 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at E2 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well F2 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-237.023 Y-11.790              ; Move XY to F2 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at F2 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well A3 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-223.886 Y53.250              ; Move XY to A3 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at A3 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well B3 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-223.886 Y40.242              ; Move XY to B3 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at B3 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well C3 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-223.886 Y27.234              ; Move XY to C3 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at C3 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well D3 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-223.886 Y14.226              ; Move XY to D3 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at D3 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well E3 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-223.886 Y1.218              ; Move XY to E3 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at E3 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well F3 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-223.886 Y-11.790              ; Move XY to F3 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at F3 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well A4 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-210.749 Y53.250              ; Move XY to A4 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at A4 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well B4 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-210.749 Y40.242              ; Move XY to B4 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at B4 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well C4 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-210.749 Y27.234              ; Move XY to C4 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at C4 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well D4 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-210.749 Y14.226              ; Move XY to D4 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at D4 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well E4 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-210.749 Y1.218              ; Move XY to E4 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at E4 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well F4 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-210.749 Y-11.790              ; Move XY to F4 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at F4 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well A5 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-197.611 Y53.250              ; Move XY to A5 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at A5 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well B5 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-197.611 Y40.242              ; Move XY to B5 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at B5 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well C5 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-197.611 Y27.234              ; Move XY to C5 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at C5 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well D5 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-197.611 Y14.226              ; Move XY to D5 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at D5 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well E5 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-197.611 Y1.218              ; Move XY to E5 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at E5 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well F5 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-197.611 Y-11.790              ; Move XY to F5 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at F5 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well A6 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-184.474 Y53.250              ; Move XY to A6 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at A6 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well B6 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-184.474 Y40.242              ; Move XY to B6 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at B6 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well C6 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-184.474 Y27.234              ; Move XY to C6 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at C6 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well D6 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-184.474 Y14.226              ; Move XY to D6 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at D6 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well E6 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-184.474 Y1.218              ; Move XY to E6 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at E6 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well F6 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-184.474 Y-11.790              ; Move XY to F6 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at F6 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well A7 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-171.337 Y53.250              ; Move XY to A7 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at A7 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well B7 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-171.337 Y40.242              ; Move XY to B7 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at B7 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well C7 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-171.337 Y27.234              ; Move XY to C7 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at C7 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well D7 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-171.337 Y14.226              ; Move XY to D7 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at D7 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well E7 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-171.337 Y1.218              ; Move XY to E7 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at E7 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well F7 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-171.337 Y-11.790              ; Move XY to F7 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at F7 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well A8 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-158.200 Y53.250              ; Move XY to A8 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at A8 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well B8 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-158.200 Y40.242              ; Move XY to B8 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at B8 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well C8 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-158.200 Y27.234              ; Move XY to C8 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at C8 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well D8 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-158.200 Y14.226              ; Move XY to D8 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at D8 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well E8 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-158.200 Y1.218              ; Move XY to E8 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at E8 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Well F8 ---
G00 G54 G90 Z40.000          ; Raise to safe Z first
G00 X-158.200 Y-11.790              ; Move XY to F8 camera position
G00 Z20.000                ; Lower to imaging height
V.E.UserInteraction.Message = "Camera at F8 - trigger imaging, then click OK"
M121                       ; Pause for manual camera trigger

; --- Return home ---
G00 G54 G90 Z40.000          ; Safe Z before going home
G800                       ; Go home
M110=1000
M30
