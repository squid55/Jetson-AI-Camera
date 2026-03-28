"""
Zybo UDP 수신 테스트 (PC에서도 실행 가능)
Zybo에서 보내는 CNN 결과 패킷을 수신해서 출력

Usage:
    python3 scripts/test_zybo_recv.py
    python3 scripts/test_zybo_recv.py --port 5001
"""

import argparse
import socket
import struct
import time


HEADER = 0xAA
STRUCT_FMT = "<BBHIB3x"  # 12 bytes total


def main():
    parser = argparse.ArgumentParser(description="Zybo CNN Result Receiver Test")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(5.0)

    print(f"Listening on UDP port {args.port}...")
    print("Waiting for Zybo CNN packets (Ctrl+C to stop)\n")

    count = 0
    while True:
        try:
            data, addr = sock.recvfrom(64)
        except socket.timeout:
            print("  (no data received in 5s)")
            continue
        except KeyboardInterrupt:
            break

        if len(data) < 12:
            print(f"  Short packet: {len(data)} bytes from {addr}")
            continue

        header, digit, conf_raw, ts, status = struct.unpack(STRUCT_FMT, data[:12])

        if header != HEADER:
            print(f"  Bad header: 0x{header:02X} from {addr}")
            continue

        count += 1
        conf_pct = conf_raw / 1023.0 * 100
        valid = "VALID" if (status & 0x01) else "INVALID"
        print(f"  [{count:5d}] Digit={digit}  Conf={conf_pct:5.1f}%  ts={ts}ms  {valid}  from {addr[0]}")

    print(f"\nTotal received: {count} packets")
    sock.close()


if __name__ == "__main__":
    main()
