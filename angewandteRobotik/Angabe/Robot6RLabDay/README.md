# Praktikum Angewandte Robotik, Teil I: 6R-Roboter (Labortag)

Aufgabenstellung der Durchführung im Labor: `labRobot6R.pdf`. Die Vorbereitung (`Robot6RPreparation.zip`,
`preparationRobot6R.pdf`) muss vor dem Labortermin abgeschlossen sein; sie wird im Labor überprüft.

Mitbringen (z.B. auf einem USB-Stick):
- `preparation14CupManipulation.py` aus der Vorbereitung (Aufgabe 1.4; ein Programm pro Gruppe) und als Reserve die
  damit erzeugte Matrix Q `solution/dataLab.txt`
- für die freiwillige Zusatzaufgabe gegebenenfalls Ihr Programm des eigenen Bewegungsablaufs, das
  `solution/dataOwn.txt` neben sich schreibt, oder nur `dataOwn.txt` (in der Simulation geprüft)

Im Labor:
- Am Laborrechner hat jede Gruppe einen eigenen Ordner; die Praktikumsleitung legt Ihre Dateien dort ab.
- Ihr Programm wird dort ausgeführt und erzeugt die Matrix Q. Jede Bewegung wird zuerst mit dem Simulationsmodell
  `robot6R` der Vorbereitung simuliert und erst danach an den realen Roboter gesendet (Aufgaben 1 bis 3 des
  Aufgabenblatts). Anpassungen (Posen, Geschwindigkeiten, Greifbefehle) machen Sie direkt in Ihrem Programm.
- Nach dem Labortermin erhält die Gruppe `gruppeXX.zip` mit den im Labor verwendeten Dateien (`qMatrix/` und
  `programs/`, gegebenenfalls im Labor angepasst) für das Protokoll.

Abgabe (ein Protokoll für beide Teilaufgaben, bis 14. Jänner 2027, 23:00 Uhr auf OLAT, genau zwei Dateien):
- `Nachname_Matrikelnummer_Protokoll.pdf`: zu jeder Aufgabe der Vorbereitung und der Durchführung die im Aufgabenblatt
  unter "Protokoll" genannten Punkte
- `Nachname_Matrikelnummer_Code.zip`: `preparation11JointAngles.py` bis `preparation14CupManipulation.py`,
  `solution/dataLab.txt` (Teil I) und `preparationStateMachine.py` (Teil II), ohne die Ordner `robot6R`, `fahrzeug`,
  `labRobots`
