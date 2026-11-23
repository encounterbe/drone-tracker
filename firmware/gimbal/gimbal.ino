#include <ESP32Servo.h>

static const uint32_t BAUD = 115200;
static const uint8_t PIN_PAN = 13;
static const uint8_t PIN_TILT = 12;

static const int PAN_MIN_US = 1000;
static const int PAN_MAX_US = 2000;
static const int TILT_MIN_US = 1200;
static const int TILT_MAX_US = 1900;

static const int PAN_CENTER_US = 1500;
static const int TILT_CENTER_US = 1550;

static const uint32_t FAILSAFE_MS = 1000;
static const uint32_t STATUS_PERIOD_MS = 250;
static const size_t RX_MAX = 48;
static const int SLEW_US_PER_TICK = 12;

Servo panServo;
Servo tiltServo;

static char rxBuf[RX_MAX];
static size_t rxLen = 0;

static int panTargetUs = PAN_CENTER_US;
static int tiltTargetUs = TILT_CENTER_US;
static int panCurrentUs = PAN_CENTER_US;
static int tiltCurrentUs = TILT_CENTER_US;

static uint32_t lastCommandMs = 0;
static uint32_t lastStatusMs = 0;
static bool failsafeActive = true;

static int clampInt(int value, int lo, int hi) {
  if (value < lo) return lo;
  if (value > hi) return hi;
  return value;
}

static int axisToMicros(long axis, int loUs, int hiUs) {
  axis = clampInt((int)axis, -1000, 1000);
  int centre = (loUs + hiUs) / 2;
  int span = (hiUs - loUs) / 2;
  return clampInt(centre + (int)((axis * span) / 1000), loUs, hiUs);
}

static int slewToward(int current, int target) {
  int delta = target - current;
  if (delta > SLEW_US_PER_TICK) delta = SLEW_US_PER_TICK;
  if (delta < -SLEW_US_PER_TICK) delta = -SLEW_US_PER_TICK;
  return current + delta;
}

static void enterFailsafe() {
  if (!failsafeActive) {
    failsafeActive = true;
    Serial.println(F("EVT FAILSAFE"));
  }
  panTargetUs = PAN_CENTER_US;
  tiltTargetUs = TILT_CENTER_US;
}

static bool parseLine(const char *line, long *pan, long *tilt, long *laser) {
  char *end = NULL;
  *pan = strtol(line, &end, 10);
  if (end == line || *end != ',') return false;

  const char *second = end + 1;
  *tilt = strtol(second, &end, 10);
  if (end == second || *end != ',') return false;

  const char *third = end + 1;
  *laser = strtol(third, &end, 10);
  if (end == third) return false;

  return true;
}

static void handleLine(const char *line) {
  if (line[0] == '\0') return;

  if (strcmp(line, "PING") == 0) {
    Serial.println(F("PONG"));
    lastCommandMs = millis();
    return;
  }

  if (strcmp(line, "STOP") == 0) {
    enterFailsafe();
    Serial.println(F("OK STOP"));
    return;
  }

  long pan = 0, tilt = 0, laser = 0;
  if (!parseLine(line, &pan, &tilt, &laser)) {
    Serial.println(F("ERR PARSE"));
    return;
  }

  if (laser != 0) {
    Serial.println(F("ERR LASER_NOT_WIRED"));
  }

  panTargetUs = axisToMicros(pan, PAN_MIN_US, PAN_MAX_US);
  tiltTargetUs = axisToMicros(tilt, TILT_MIN_US, TILT_MAX_US);
  lastCommandMs = millis();

  if (failsafeActive) {
    failsafeActive = false;
    Serial.println(F("EVT ACTIVE"));
  }
}

static void pollSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      rxBuf[rxLen] = '\0';
      handleLine(rxBuf);
      rxLen = 0;
      continue;
    }
    if (rxLen < RX_MAX - 1) {
      rxBuf[rxLen++] = c;
    } else {
      rxLen = 0;
      Serial.println(F("ERR OVERFLOW"));
    }
  }
}

static void emitStatus() {
  Serial.print(F("STA "));
  Serial.print(panCurrentUs);
  Serial.print(',');
  Serial.print(tiltCurrentUs);
  Serial.print(',');
  Serial.println(failsafeActive ? 1 : 0);
}

void setup() {
  Serial.begin(BAUD);

  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  panServo.attach(PIN_PAN, PAN_MIN_US, PAN_MAX_US);
  tiltServo.attach(PIN_TILT, TILT_MIN_US, TILT_MAX_US);

  panServo.writeMicroseconds(panCurrentUs);
  tiltServo.writeMicroseconds(tiltCurrentUs);

  lastCommandMs = millis();
  Serial.println(F("EVT READY gimbal-1 proto=pan,tilt,laser"));
}

void loop() {
  pollSerial();

  uint32_t now = millis();
  if (now - lastCommandMs > FAILSAFE_MS) {
    enterFailsafe();
  }

  panCurrentUs = slewToward(panCurrentUs, panTargetUs);
  tiltCurrentUs = slewToward(tiltCurrentUs, tiltTargetUs);
  panServo.writeMicroseconds(panCurrentUs);
  tiltServo.writeMicroseconds(tiltCurrentUs);

  if (now - lastStatusMs >= STATUS_PERIOD_MS) {
    lastStatusMs = now;
    emitStatus();
  }

  delay(10);
}
