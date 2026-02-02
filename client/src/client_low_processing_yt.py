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
    timing_breakdown = {'network': [], 'yolo': [], 'client': []}
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
            if frame_count == tot_frame or int(time.time() - start_time) > duration:
                raise KeyboardInterrupt
            
            # Timing rete + YOLO
            before_request = time.time()
            # Send frame to the server
            files = {'frames': ('frame.jpg', frame_bytes.getvalue(), 'image/jpeg')}
            response = requests.post(url, files=files)
            network_time = time.time() - before_request
            
            response_data = response.json()
            yolo_time = response_data[0]['yolo_processing_time']
            detections = response_data[0]['detections']
            
            # Timing client-side
            before_client = time.time()
            
            # Draw boxes only every X frames
            if frame_count % 10000 == 0:
                for bbox in detections:
                    cv2.rectangle(compressed_frame, (bbox['xmin'], bbox['ymin']), 
                                  (bbox['xmax'], bbox['ymax']), (0, 255, 0), 2)
                    cv2.putText(compressed_frame, f"{bbox['name']} {bbox['confidence']:.2f}", 
                               (bbox['xmin'], bbox['ymin'] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imwrite(output_image_path, compressed_frame)
            
            client_time = time.time() - before_client
            
            # Record timing breakdown
            timing_breakdown['network'].append(network_time)
            timing_breakdown['yolo'].append(yolo_time)
            timing_breakdown['client'].append(client_time)
            
            frame_count += 1
            
            # Count framerate
            after = time.time()
            if str(round(after)) not in stats:
                stats[str(round(after))] = 0
            stats[str(round(after))] += 1
            
            elapsed_time = network_time + yolo_time + client_time
            sleep_time = max(0, interval - elapsed_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n=== Performance Breakdown ===")
        print(f"Avg Network latency: {np.mean(timing_breakdown['network']):.3f}s")
        print(f"Avg YOLO processing: {np.mean(timing_breakdown['yolo']):.3f}s")
        print(f"Avg Client processing: {np.mean(timing_breakdown['client']):.3f}s")
        print(f"Total average: {np.mean(timing_breakdown['network']) + np.mean(timing_breakdown['yolo']) + np.mean(timing_breakdown['client']):.3f}s")
        
        tot_f = sum(stats.values())
        count = len(stats)
        measured_fps = tot_f / count if count > 0 else 0
        print(f"\nExpected FPS: {framerate}")
        print(f"Measured FPS: {round(measured_fps, 3)}")
        print(f"Difference: {round(((framerate - measured_fps) / framerate) * 100, 2)}%")
        
        with open("../data/result.dat", 'a') as file1:
            file1.write(f"{datetime.datetime.now().time()},{framerate},{round(measured_fps, 3)}\n")

        with open("../data/stats.dat", 'w') as stats_file:
            stats_file.write("Timestamp,Frame count\n")
            for key in stats:
                stats_file.write(f"{key},{stats[key]}\n")

if __name__ == "__main__":
    
    check_and_delete_file("../data/result.dat")
    #for i in range(1, 11):
    process_image_repeatedly(1, 1, 'http://<ip>:<port>/process_frames')
        #time.sleep(60)