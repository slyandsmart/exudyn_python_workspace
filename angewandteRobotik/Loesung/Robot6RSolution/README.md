# Lösung (nur für die Praktikumsleitung, nicht an die Studierenden weitergeben)

Wie das Studentenpaket, aber mit den vollständigen Dateien der Vorbereitung (ohne `## TODO`) und der Musterlösung im
Unterordner `reference` des Modells. Die Dateien laufen wie im Studentenpaket beschrieben.

# Praktikum Angewandte Robotik, Teil I: 6R-Roboter (Vorbereitung)

Aufgabenstellung: `preparationRobot6R.pdf` (mit einer freiwilligen Zusatzaufgabe: ein eigener Bewegungsablauf,
zuerst simuliert, dann im Labor am realen Roboter). Worum es geht: `videoRobot6R.mp4` (der 6R-Roboter bringt den Becher
von Tisch 1 auf Tisch 2, Simulation im Labor).

1. Anaconda Prompt, Umgebung der LV aktivieren (`conda activate robotik`, Python 3.13), in den Ordner dieses Pakets
   wechseln und `pip install -r requirements.txt` ausführen. Das installiert Exudyn aus dem Ordner `wheels` (die
   Version auf PyPI reicht nicht), auch wenn schon eine ältere Exudyn-Version installiert ist. Starten Sie die Dateien
   immer in diesem Ordner (Spyder: Arbeitsverzeichnis = Ordner dieses Pakets), sonst werden die Module des Pakets
   nicht gefunden.
2. Ergänzen Sie die mit `## TODO` markierten Stellen der Vorlagen:
   - `preparation11JointAngles.py` (Aufgabe 1.1, Gelenkwinkel vorgeben und fahren)
   - `preparation12ForwardKinematics.py` (Aufgabe 1.2, DH-Parameter)
   - `preparation13Configurations.py` (Aufgabe 1.3, Roboterkonfigurationen: elbow up und elbow down wie beim
     3R-Roboter der Hausübung, eine Demonstration ohne TODO, die Ergebnisse werden im Protokoll gedeutet)
   - `preparation14CupManipulation.py` (Aufgabe 1.4, Greifposen und Becher mit Geschwindigkeiten und Greifbefehlen,
     speichert solution/dataLab.txt)
   - `preparation15PTPMotion.py` (Aufgabe 1.5, PTP-Bewegungsarten und Überschleifen: eine Demonstration ohne TODO,
     die Ergebnisse werden im Protokoll gedeutet)
3. Starten: `python preparation11JointAngles.py` (ohne Visualisierung: `--noGraphics`); jede Vorlage läuft für sich,
   auch wenn die anderen noch nicht gelöst sind. Oben in jeder Vorlage schalten `True`/`False` ihre Teilaufgaben ein
   oder aus; der Renderer zeigt die Ergebnisse (Koordinatensysteme der Posen, die Bewegung des Roboters).

Ins Labor bringen Sie Ihr Programm `preparation14CupManipulation.py` (Aufgabe 1.4) und als Reserve die damit erzeugte
Datei `solution/dataLab.txt` mit, z.B. auf einem USB-Stick. Im Labor wird Ihr Programm ausgeführt; Anpassungen machen
Sie dort direkt in Ihrem Programm.

Abgabe (ein Protokoll für beide Teilaufgaben, bis 14. Jänner 2027, 23:00 Uhr auf OLAT, genau zwei Dateien):
- `Nachname_Matrikelnummer_Protokoll.pdf`: zu jeder Aufgabe der Vorbereitung und der Durchführung die im Aufgabenblatt
  unter "Protokoll" genannten Punkte (u.a. die ausgedruckte und ergänzte Abbildung 4)
- `Nachname_Matrikelnummer_Code.zip`: `preparation11JointAngles.py` bis `preparation14CupManipulation.py`,
  `solution/dataLab.txt` (Teil I) und `preparationStateMachine.py` (Teil II), ohne die Ordner `robot6R`, `fahrzeug`,
  `labRobots`

Die Ordner `robot6R` (Simulation der Aufgabe, Exudyn) und `labRobots` (Gerätedaten und Modell des Laborroboters)
müssen nicht verändert werden.
