import cv2
import requests
import time
from io import BytesIO
import datetime
import numpy as np
import os

def check_and_delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")
    else:
        print(f"File '{file_path}' does not exist.")
        
    with open("../data/result.dat", 'a') as file1:
        file1.write(f"Timestamp,Expected framerate,Measured framerate\n")

def process_image_repeatedly(framerate, duration, url):
    """
    Process an image repeatedly by sending frames to a server and saving the processed compressed image.
    Args:
        framerate (int): The desired framerate (frames per second).
        duration (int): The duration of the processing in seconds.
    Returns:
        None
    Raises:
        KeyboardInterrupt: If the processing is interrupted by the user.
    """
    stats = {}
    start_time = time.time()

    # Replace with your server URL
    #url = 'http://192.168.11.90:32025/process_frames'
    url = 'http://localhost:5000/process_frames'

    # Path to the image file
    image_path = '../img/dog_bike_car.jpg'

    # Output image path
    output_image_path = '../img/compressed_image.jpg'

    # Configurable rate (requests per second)
    rate = framerate  # Example: 1 request per second
    interval = 1 / rate
    tot_frame = framerate * duration

    # Compression quality (0-100), higher value means better quality but larger size
    compression_quality = 10  # Adjust as needed

    # Read the image
    frame = cv2.imread(image_path)

    if frame is None:
        print("Failed to read the image.")
        return

    # Compress frame to JPEG format in memory
    _, frame_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])
    frame_bytes = BytesIO(frame_encoded)

    # Convert compressed bytes back to an image for processing
    compressed_frame = cv2.imdecode(np.frombuffer(frame_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    frame_count = 0
    try:
        while True:
            if frame_count == tot_frame:
                raise KeyboardInterrupt
            
            before = time.time()

            if int(before - start_time) > duration:
                raise KeyboardInterrupt

            # Send frame to the server
            files = {'frames': ('frame.jpg', frame_bytes.getvalue(), 'image/jpeg')}
            response = requests.post(url, files=files)
            print(f"Response: {response.text}")
            bounding_boxes = response.json()
            frame_count += 1

            # Draw bounding boxes on the compressed frame
            for bbox in bounding_boxes[0]:
                cv2.rectangle(compressed_frame, (bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax']), (0, 255, 0), 2)
                cv2.putText(compressed_frame, f"{bbox['name']} {bbox['confidence']:.2f}", (bbox['xmin'], bbox['ymin'] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the processed compressed image
            cv2.imwrite(output_image_path, compressed_frame)

            after = time.time()
            if str(round(after)) not in stats:
                stats[str(round(after))] = 0 
            stats[str(round(after))] += 1 

            elapsed_time = after - before
            sleep_time = max(0, interval - elapsed_time)

            # Wait for the next interval
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("End processing.")
        # Print stats
        print("Stats:", stats)
        tot_f = 0
        count = 0
        for key in stats:
            tot_f += stats[key]
            count += 1
        print(f"Expected framerate: {framerate}\t Measured framerate: {round(tot_f/count, 3)}")
        # Stampa anche la percentuale della differenza tra espected e measured
        print(f"Difference: {round(((framerate - (tot_f/count)) / framerate) * 100, 2)}%")
        
        with open("../data/result.dat", 'a') as file1:
            file1.write(f"{datetime.datetime.now().time()},{framerate},{round(tot_f/count, 3)}\n")

if __name__ == "__main__":
    
    check_and_delete_file("../data/result.dat")
    #for i in range(1, 11):
    process_image_repeatedly(24, 5, 'http://<ip>:<port>/process_frames')
        #time.sleep(60)