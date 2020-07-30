FROM  centos
RUN yum install python36 -y
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow
RUN pip3 install keras
RUN pip3 install pillow 
RUN pip3 install opencv-python
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install matplotlib
WORKDIR /root/task3mlops/
COPY main_cnn.py /root/task3mlops/

CMD ["python36", "main_cnn.py"]

