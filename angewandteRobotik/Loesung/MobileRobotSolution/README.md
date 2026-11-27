# Lösung (nur für die Praktikumsleitung, nicht an die Studierenden weitergeben)

Wie das Studentenpaket, aber mit den vollständigen Dateien der Vorbereitung (ohne `## TODO`) und der Musterlösung im
Unterordner `reference` des Modells. Die Dateien laufen wie im Studentenpaket beschrieben.

# Praktikum Angewandte Robotik, Teil II: Mobile Robotik (Vorbereitung)

Aufgabenstellung: `preparationMobileRobot.pdf` (mit einer freiwilligen Zusatzaufgabe: ein eigener Ablauf des
Fahrzeugs, zuerst simuliert, dann im Labor am realen Fahrzeug). Worum es geht: `videoKairos.mp4` (der Kairos fährt zum
Tisch, greift das Objekt und kehrt zurück, Simulation im Labor).

1. Anaconda Prompt, Umgebung der LV aktivieren (`conda activate robotik`, Python 3.13), in den Ordner dieses Pakets
   wechseln und `pip install -r requirements.txt` ausführen. Das installiert Exudyn aus dem Ordner `wheels` (die
   Version auf PyPI reicht nicht), auch wenn schon eine ältere Exudyn-Version installiert ist. Starten Sie die Dateien
   immer in diesem Ordner (Spyder: Arbeitsverzeichnis = Ordner dieses Pakets), sonst werden die Module des Pakets
   nicht gefunden.
2. Ergänzen Sie in `preparationStateMachine.py` die mit `## TODO` markierten Hilfsfunktionen (`ApproachPose`,
   `DriveCommand`, `ArmBasePose`, `GraspPoses`) und den Zustandsautomaten `functionStateMachine`.
3. Starten: `python preparationStateMachine.py` oder `python runSimulation.py`
   (Optionen: `--noGraphics`, `--initialPose X Y YAW`, `--realtime`, `python runSimulation.py --help`).
   Der Renderer zeigt die Simulation live, danach spielt der SolutionViewer sie erneut ab.
4. Teilaufgaben einzeln: `runTask` oben in der Vorlage (`'drive'`, `'grasp'`, `'return'`, `'stateMachine'`);
   `showApproachPose`, `showArmBase`, `showGraspPoses` zeigen die Ergebnisse als Koordinatensysteme im Renderer.

Im Labor wird dieselbe Datei `preparationStateMachine.py` mit dem realen Roboter verwendet (mit `runTask = 'stateMachine'`);
bringen Sie nur diese Datei ins Labor mit, z.B. auf einem USB-Stick.

Abgabe (ein Protokoll für beide Teilaufgaben, bis 14. Jänner 2027, 23:00 Uhr auf OLAT, genau zwei Dateien):
- `Nachname_Matrikelnummer_Protokoll.pdf`: zu jeder Aufgabe der Vorbereitung und der Durchführung die im Aufgabenblatt
  unter "Protokoll" genannten Punkte (u.a. die Skizze des Zustandsautomaten)
- `Nachname_Matrikelnummer_Code.zip`: `preparation11JointAngles.py` bis `preparation14CupManipulation.py`,
  `solution/dataLab.txt` (Teil I) und `preparationStateMachine.py` in der im Labor zuletzt verwendeten Fassung (Teil II),
  ohne die Ordner `robot6R`, `fahrzeug`, `labRobots`
Die Ordner `fahrzeug` (Simulationsmodell, Exudyn) und `labRobots` (Gerätedaten, Kollisionsprüfung und Bahnplanung des
Arms) müssen nicht verändert werden.
