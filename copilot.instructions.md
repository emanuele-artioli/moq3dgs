## **description: 'Master Agent System Instructions for 3DGS-MoQ: Viewport-Aware 3D Gaussian Splatting over Media-over-QUIC' applyTo: '/*.py, //*.cpp, \*\*/environment.yaml, \*\*/pyproject.toml, /README.md'**

# **3DGS-MoQ System Development**

## **Instructions (Highest Priority)**

* **Permissions:** You are NOT an admin. DO NOT run commands with sudo or apt.  
* **Headless Mode:** Assume a headless multi-GPU Linux environment (CUDA). DO NOT run graphical applications (e.g., cv2.imshow(), plt.show(), openGL windows). **Instead of displaying graphical interfaces, always save renders, metrics, and network logs to the disk.**  
* **Dependency Management:** pyproject.toml is the ONE AND ONLY source of truth for standard Python packages. If you need a new pip package, add it there. Use environment.yaml STRICTLY as a GPU environment bootstrapper for heavy CUDA drivers, PyTorch binaries, and customized 3DGS rasterizer wheels. Never use raw requirements.txt.  
* **Execution Environment:** Check for the existence of the 3dgs\_moq conda environment and run all scripts within it.  
* **Default Test File:** Unless specified otherwise, always use the default 3DGS ply dataset for testing and runs: /home/itec/emanuele/3dgs\_moq/assets/train\_scene/point\_cloud.ply and the associated movement trace /home/itec/emanuele/3dgs\_moq/assets/traces/eval\_trace\_01.json.  
* **Real Experiment Input Requirement:** Any real experiment (non-synthetic evaluation intended to validate end-to-end network behavior) MUST pass \--scene /home/itec/emanuele/3dgs\_moq/assets/train\_scene/ explicitly. Do not rely on mock data fallbacks for networking tests.  
* **Testing & Review Workflow:** Use test-driven generation. Always write a unit test or integration script *before* or *alongside* new core logic (e.g., MoQ packetization, spatial clustering, frustum culling).

## **Architecture Best Practices**

### **1\. MoQ Transport Mapping (The Protocol)**

Media over QUIC (MoQ) operates on a strict hierarchy. Ensure all data structures adhere to this mapping:

* **Broadcast:** The entire 3DGS scene (e.g., "Train Scene").  
* **Track:** A distinct Spatial Volume (Octree node) or Semantic Region.  
* **Group:** A specific chunk of Gaussian clusters within that volume.  
* **Object (Subgroup):** The Level of Detail (LoD) or bitrate ladder.  
  * *Object 0:* Base geometry \+ Opacity \+ DC color (SH0).  
  * *Object 1:* High-frequency details (SH1-SH3) \+ Scale refinements.

### **2\. State & Caching**

* **Do not resend splats.** The client must maintain a local persistent buffer of received splat clusters.  
* The server provides a "Manifest" mapping spatial coordinates to MoQ Track/Group IDs.  
* The client calculates the view frustum locally, compares it against its cached chunks, and issues MoQ SUBSCRIBE updates for missing or low-quality chunks.

### **3\. Dynamic Prioritization**

* Priority ranges from 0 (Highest/Critical) to 255 (Lowest/Droppable).  
* Always calculate priority based on:  
  1. **Distance to camera:** Closer \= higher priority.  
  2. **Frustum position:** Center screen \= higher priority. Periphery \= medium. Occluded/Behind \= lowest.  
  3. **Layer type:** Base LoD (Object 0\) \= higher priority than Enhancement LoD (Object 1).

## **Code Style & Structure**

### **Strict Typing and Docstrings**

Enforce Python 3.10+ typing strictly. Every function must have clear input/output types and a docstring explaining the *why*.  
from typing import Dict, List, Tuple  
import torch

def cluster\_gaussians(  
    means3D: torch.Tensor,   
    num\_clusters: int  
) \-\> Dict\[int, List\[int\]\]:  
    """  
    Partitions global 3DGS points into spatial clusters for MoQ Track assignment.  
      
    Args:  
        means3D: \[N, 3\] tensor of Gaussian centers.  
        num\_clusters: Target number of spatial volumes.  
          
    Returns:  
        Dict mapping cluster\_id to a list of original Gaussian indices.  
    """  
    pass

### **Resource Tagging (The DAG)**

Use decorators to clearly separate CPU/Network-bound tasks from GPU-bound rendering/processing tasks.  
from typing import Callable

def network\_bound(func: Callable):  
    """Tag for QUIC transport, MoQ packetization, and socket I/O"""  
    func.is\_network\_bound \= True  
    return func

def gpu\_bound(func: Callable):  
    """Tag for PyTorch inference, rasterization, and tensor math"""  
    func.is\_gpu\_bound \= True  
    return func

### **Data Contracts with Pydantic**

Use Pydantic for all Manifests, Frustum updates, and Client-Server messaging.  
from pydantic import BaseModel  
from typing import Literal, List, Optional

class ViewportUpdate(BaseModel):  
    """Client message requesting new spatial data based on movement trace."""  
    client\_id: str  
    timestamp\_ms: int  
    camera\_position: List\[float\] \# \[x, y, z\]  
    view\_matrix: List\[List\[float\]\]  
    fov: float

class MoQSubscription(BaseModel):  
    track\_id: str  
    group\_id: str  
    max\_object\_id: int  \# Defines the requested Quality Level  
    priority: int       \# 0-255 dynamically assigned by client/server

## **Important Reminders**

* **File Generation:** If modifying C++/CUDA rasterizer code or generating long Python classes, output the code inside proper file blocks.  
* **Artifact Mitigation:** When handling Level-of-Detail transitions between spatial chunks, ensure the base track (low-res global scene) is always prioritized to prevent "dumb volume" hard-edge artifacts.  
* **Modularity:** Keep the MoQ transport layer completely decoupled from the 3DGS rendering layer. They should communicate via asynchronous queues or shared memory buffers.