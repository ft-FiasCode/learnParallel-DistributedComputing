import pyopencl as cl

def main():
    for platform in cl.get_platforms():
        print(f"\nPlatform: {platform.name} \n  Vendor: {platform.vendor} \n Version: {platform.version}")
        for device in platform.get_devices():
            dtype = cl.device_type.to_string(device.type)
            print(f"  Device: {device.name} \n Type: {dtype} \n Version: {device.version}")

if __name__ == "__main__":
    main()
