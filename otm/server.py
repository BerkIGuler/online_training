import socket
import os
import zipfile
import tarfile


class TarDir:
    def __init__(self, tar_file_name, dir_path):
        self.tar_file_name = tar_file_name
        self.dir_path = dir_path

    def tardir(self):
        with tarfile.open(self.tar_file_name, "w:gz") as tar:
            tar.add(self.dir_path, arcname=os.path.basename(self.dir_path))


class ZipDir:
    def __init__(self, zip_file_name, dir_path):
        self.zip_file_name = zip_file_name
        self.dir_path = dir_path

    def zipdir(self):
        zipf = zipfile.ZipFile(self.zip_file_name, 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(self.dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path)
        zipf.close()


def receive(file_name, client_socket):
    with open(file_name, 'wb') as f:
        while True:
            chunk = client_socket.recv(10240)
            if not chunk:
                print("receive done")
                break
            if chunk.endswith(b"Sent Done"):
                chunk = chunk[:-9]
                f.write(chunk)
                break
            else:
                f.write(chunk)
        print("Finish")


def send(folder_path, client_socket):
    zipper = ZipDir("model.zip", folder_path)
    zipper.zipdir()
    with open("model.zip", 'rb') as f:
        file_data = f.read()
    client_socket.sendall(file_data)
    client_socket.send(b"Sent Done")
    print('file sent to server')
    os.remove("model.zip")


class TCPServer:

    def __init__(self, host, port, folder_path, receive_file_name):
        self.host = host
        self.port = port
        self.folder_path = folder_path
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.receive_file_name = receive_file_name

    def start(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(self.folder_path)
        print(f"Server listening on {self.host}:{self.port}")

    def serve(self):
        client_socket, client_address = self.server_socket.accept()
        print(f'Client {client_address} connected')
        data = client_socket.recv(10240)

        if data.decode() == 'r':
            receive(self.receive_file_name, client_socket)
            return "r"

        elif data.decode() == 's':
            send(self.folder_path, client_socket)

        else:
            print('Invalid input received from client')

    def close(self):
        self.server_socket.close()
        print("Server closed.")


def main(HOST, PORT, folder_path, receive_file_name):

    server = TCPServer(HOST, PORT, folder_path, receive_file_name)
    server.start()
    server.serve()
    server.close()


if __name__ == '__main__':
    main()
