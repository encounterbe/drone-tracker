# ESP32 pan/tilt gimbal firmware

Drives two hobby servos from serial commands produced by
`src/control/gimbal.py`.

> This firmware is a **reimplementation** from the protocol specification, not a
> recovered original. It matches the documented line protocol and failsafe
> behaviour, but it has not been run against hardware — bench-test it with the
> servos unloaded before mounting anything.

## Build

Arduino IDE or `arduino-cli`. Board: any ESP32 dev module. One library:

* **ESP32Servo** (Library Manager → "ESP32Servo" by Kevin Harrington)

```
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/gimbal
arduino-cli upload  --fqbn esp32:esp32:esp32 -p COM3 firmware/gimbal
```

## Wiring

| Signal | Pin | Note |
|---|---|---|
| Pan servo | GPIO 13 | |
| Tilt servo | GPIO 12 | |
| Servo V+ | **external 5–6 V supply** | not the ESP32 3V3 rail |
| Ground | common | ESP32 ground must be tied to the supply ground |

Servos draw stall currents that a USB port will not deliver; powering them from
the board browns it out mid-move, which looks exactly like a firmware bug.

## Protocol

115200 baud, 8N1, newline-terminated ASCII.

**Host → device**

| Command | Meaning |
|---|---|
| `<pan>,<tilt>,<laser>` | axis demands in `[-1000, 1000]`; `laser` must be `0` |
| `PING` | keep-alive, answered `PONG` |
| `STOP` | centre immediately and enter failsafe |

**Device → host**

| Message | Meaning |
|---|---|
| `EVT READY gimbal-1 proto=pan,tilt,laser` | sent once at boot |
| `EVT ACTIVE` | first valid command after a failsafe |
| `EVT FAILSAFE` | no command for 1 s; servos re-centred |
| `STA <panUs>,<tiltUs>,<failsafe>` | status, every 250 ms |
| `ERR PARSE` / `ERR OVERFLOW` | malformed or over-long line |
| `ERR LASER_NOT_WIRED` | non-zero laser field; ignored |

## Safety

The **1 s failsafe lives on the device on purpose**. A crashed host, an
unplugged cable or a hung Python process must not leave the servos driving. Do
not move this check to the host side.

Tilt is limited in software to `TILT_MIN_US`/`TILT_MAX_US`. Set these to match
your mechanics *before* the first powered run — a servo commanded past the
frame's travel will stall and burn out.

Motion is rate-limited by `SLEW_US_PER_TICK` so a large demand step becomes a
smooth sweep rather than a snap that can tip a light tripod.

### The laser output

`<laser>` exists in the protocol only because the original firmware had it. It
is **not wired to anything**, and any non-zero value is rejected with
`ERR LASER_NOT_WIRED`.

This is a legal and safety question, not a technical one. A 5 mW pointer cannot
damage an aircraft, but directing a laser at one is a criminal offence in
Germany (§ 315 StGB and Luftverkehrsgesetz provisions) — and a tracker that is
slightly off, or that locks onto a manned aircraft instead of the intended
drone, points it somewhere nobody chose. The defensible build drives the camera,
keeps the target framed and logs the track.

Raise it with the project owner rather than quietly connecting the lock signal
to a laser driver.
