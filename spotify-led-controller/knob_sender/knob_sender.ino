/*
 * Knob Sender + NeoPixel driver
 *
 * Bidirectional serial:
 *   OUT → K<0-1023>\n          knob reading at ~50 Hz
 *   IN  ← L<r1>,<g1>,<b1>,<r2>,<g2>,<b2>,<split>\n
 *          Sets the first <split> pixels to color1, the rest to color2.
 *
 * Wiring:
 *   Potentiometer outer pins → 5V and GND
 *   Potentiometer wiper      → A0
 *   NeoPixel DIN              → pin 12
 *   NeoPixel VCC/GND          → 5V / GND
 */

#include <Adafruit_NeoPixel.h>

#define KNOB_PIN       A0
#define LED_PIN        12
#define NUM_PIXELS     8
#define SEND_INTERVAL  20     // ms (~50 Hz)
#define SMOOTHING      0.15f

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

float    smoothed = 0;
unsigned long lastSend = 0;
char     cmdBuf[64];
uint8_t  cmdLen = 0;

/* ---------- helpers ---------- */

int parseInts(const char *str, int *out, int maxCount) {
  int count = 0;
  const char *p = str;
  while (*p && count < maxCount) {
    out[count++] = atoi(p);
    while (*p && *p != ',') p++;
    if (*p == ',') p++;
  }
  return count;
}

void handleCommand() {
  if (cmdBuf[0] != 'L') return;

  // L<r1>,<g1>,<b1>,<r2>,<g2>,<b2>,<split>
  int v[7];
  if (parseInts(cmdBuf + 1, v, 7) != 7) return;

  int split = constrain(v[6], 0, NUM_PIXELS);
  for (int i = 0; i < NUM_PIXELS; i++) {
    if (i < split)
      strip.setPixelColor(i, strip.Color(v[0], v[1], v[2]));
    else
      strip.setPixelColor(i, strip.Color(v[3], v[4], v[5]));
  }
  strip.show();
}

/* ---------- main ---------- */

void setup() {
  Serial.begin(9600);
  smoothed = analogRead(KNOB_PIN);
  strip.begin();
  strip.clear();
  strip.show();
}

void loop() {
  // --- send knob value ---
  int raw = analogRead(KNOB_PIN);
  smoothed += SMOOTHING * (raw - smoothed);

  unsigned long now = millis();
  if (now - lastSend >= SEND_INTERVAL) {
    Serial.print("K");
    Serial.println((int)smoothed);
    lastSend = now;
  }

  // --- receive LED commands ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      cmdBuf[cmdLen] = '\0';
      handleCommand();
      cmdLen = 0;
    } else if (cmdLen < sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = c;
    }
  }
}
