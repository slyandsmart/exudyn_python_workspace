# Praktikum Angewandte Robotik, Teil II: Mobile Robotik (Labortag)

Aufgabenstellung der Durchführung im Labor: `labMobileRobot.pdf`. Die Vorbereitung (`MobileRobotPreparation.zip`,
`preparationMobileRobot.pdf`) muss vor dem Labortermin abgeschlossen sein; sie wird im Labor überprüft.

Mitbringen (z.B. auf einem USB-Stick):
- `preparationStateMachine.py` aus der Vorbereitung, lauffähig in der Simulation (`python runSimulation.py`)
- für die freiwillige Zusatzaufgabe gegebenenfalls Ihre eigene Ablaufsteuerung (z.B. `ownStateMachine.py`), in der
  Simulation vollständig und ohne Kollision (`python runSimulation.py --taskFile ownStateMachine.py`)

Im Labor:
- Am Laborrechner hat jede Gruppe einen eigenen Ordner; dort wird dieselbe Datei `preparationStateMachine.py` mit dem
  realen Fahrzeug verwendet (Pose des Fahrzeugs und Position des Objekts vom Vicon-System). Sie bearbeiten Ihre Datei
  dort, sie wird bei jedem Lauf neu geladen.
- Die Läufe der Reihe nach: nur Simulation mit den Live-Daten, nur Fahrbefehle (Aufgabe 1a), nur Arm und Greifer
  (Aufgabe 1b), der ganze Ablauf (Aufgabe 2). Befehle an das Fahrzeug startet die Praktikumsleitung.
- Nach dem Labortermin erhält die Gruppe `gruppeXX.zip` für das Protokoll: die zuletzt verwendete Ablaufsteuerung und
  `measurements/`, je Lauf der simulierte Posenverlauf (`<datum-zeit>_simulation_<lauf>.txt`: Zeit, x, y,
  Gierwinkel) und die Messdaten (`vehicle.txt`: Vicon x, y, z, Gierwinkel; `joints.txt`: UR5; `gripper.txt`: Breite,
  Kraft; `drive.txt`; `events.txt`).

Abgabe (ein Protokoll für beide Teilaufgaben, bis 14. Jänner 2027, 23:00 Uhr auf OLAT, genau zwei Dateien):
- `Nachname_Matrikelnummer_Protokoll.pdf`: zu jeder Aufgabe der Vorbereitung und der Durchführung die im Aufgabenblatt
  unter "Protokoll" genannten Punkte
- `Nachname_Matrikelnummer_Code.zip`: `preparation11JointAngles.py` bis `preparation14CupManipulation.py`,
  `solution/dataLab.txt` (Teil I) und `preparationStateMachine.py` in der im Labor zuletzt verwendeten Fassung (Teil II),
  ohne die Ordner `robot6R`, `fahrzeug`, `labRobots`
