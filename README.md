# Optimized Eulerian Video Magnification

## Abstract
The use of algorithms like the [Eulerian Video Magnification (EVM)](http://people.csail.mit.edu/mrub/evm/) could make a low-cost alternative for monitoring vital signs. A remote, non-invasive patient monitoring is advantageous, and it is possible using EVM. However, computational resources may be optimized to be executed in memory and power efficient computer architectures. Current implementations of the EVM lack of parallelization and its memory management can be improved using low-level languages. Our project seeks to optimize the Magnification algorithm to detect vital signs like respiratory and heart rate, non-invasive and more efficiently. According to our tests, both execution times and memory use are improved. The obtained results show an average improvement of 434% in execution times, with a maximum speedup of 746%. In addition, the implemented algorithm utilizes 200 MB less memory in average.

## Design
### Components
![image](https://user-images.githubusercontent.com/31488944/142742294-a084ad8c-73c1-4ede-b193-9d76b3aa6472.png)

### Dependencies
<img src="https://user-images.githubusercontent.com/31488944/142742314-8529aaf0-a772-4133-ab31-bc6b7237b1ec.png" width="640">

## Results
### Execution times
<img src="https://user-images.githubusercontent.com/31488944/142742327-619e248c-312c-4b76-a796-2ad8ed201033.png" width="420">

### Memory use
![image](https://user-images.githubusercontent.com/31488944/142742335-443c3463-3a31-4705-8c62-45d62bdbd46f.png)

## More
- [Paper](https://drive.google.com/open?id=1KYjeBpGlytra5XYvsP52LJGyfW6m88bG&authuser=edmobepersonal%40gmail.com&usp=drive_fs) published in [URUCON 2021](http://urucon2021.org/accepted_papers.html)
- [Sorftware Requirements Specification document (Spanish)](https://drive.google.com/file/d/1IlCPeKQuEBswGQdrJSZE0omLrGmlNw5r/view?usp=sharing)
- [Software Design document (Spanish)](https://drive.google.com/file/d/1J5ISYAXXiNFusLBbf9aKeGKw0pK86sId/view?usp=sharing)
