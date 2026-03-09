/*
 * Knob Sender — reads a potentiometer on A0 and streams the
 * smoothed value to the host over serial.
 *
 * Protocol:  K<0-1023>\n   (ASCII, newline-terminated)
 * Rate:      ~50 Hz (every 20 ms)
 *
 * Wiring:
 *   Potentiometer outer pins → 5V and GND
 *   Potentiometer wiper      → A0
 */

const int KNOB_PIN = A0;
const unsigned long SEND_INTERVAL_MS = 20;
const float SMOOTHING = 0.15;  // exponential moving average factor

float smoothed = 0;
unsigned long lastSend = 0;

void setup() {
  Serial.begin(9600);
  smoothed = analogRead(KNOB_PIN);
}

void loop() {
  int raw = analogRead(KNOB_PIN);
  smoothed += SMOOTHING * (raw - smoothed);

  unsigned long now = millis();
  if (now - lastSend >= SEND_INTERVAL_MS) {
    Serial.print("K");
    Serial.println((int)smoothed);
    lastSend = now;
  }
}
