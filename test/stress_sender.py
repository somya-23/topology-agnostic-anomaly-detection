import socket
import time

HOST = '10.53.1.140'  # Server address
PORT = 24122        # Port to connect

def send_message(message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(message.encode())

if __name__ == "__main__":
    # Start memory leak
    print("Starting memory leak...")
    send_message("1024")

    time.sleep(50)

    # Start CPU overload
    print("Starting CPU overload...")
    send_message("5")

    time.sleep(5)

    # Stop everything
    print("Stopping stress test...")
    send_message("END")
