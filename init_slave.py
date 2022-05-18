from pyduino.slave import ReactorServer

if __name__=="__main__":
    rs = ReactorServer(import_name="Pyduino Slave Server")
    rs.run(port=5000,host="0.0.0.0")