import cv2
import numpy as np

def generate_video(filename="input.mp4", duration=5, fps=30, width=640, height=480):
    print(f"Generating dummy video: {filename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # Agents state: x, y, dx, dy
    agents = []
    for _ in range(5):
        agents.append({
            'x': np.random.randint(0, width),
            'y': np.random.randint(0, height),
            'dx': np.random.randint(-5, 5),
            'dy': np.random.randint(-5, 5),
            'color': (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        })
        
    for frame_idx in range(duration * fps):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        for agent in agents:
            # Update position
            agent['x'] += agent['dx']
            agent['y'] += agent['dy']
            
            # Bounce
            if agent['x'] < 0 or agent['x'] > width: agent['dx'] *= -1
            if agent['y'] < 0 or agent['y'] > height: agent['dy'] *= -1
            
            # Draw person (circle + rectangle)
            cv2.circle(frame, (int(agent['x']), int(agent['y'])), 10, agent['color'], -1)
            cv2.rectangle(frame, (int(agent['x'])-10, int(agent['y'])), (int(agent['x'])+10, int(agent['y'])+40), agent['color'], -1)
            
        out.write(frame)
        
    out.release()
    print("Video generated successfully.")

if __name__ == "__main__":
    generate_video()
