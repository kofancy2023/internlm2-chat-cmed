# # GitHub 根目录下的 app.py 文件会作为应用的启动的脚本，请务必在根目录下创建 app.py 的文件
# # https://zhuanlan.zhihu.com/p/679819824
#
# import os
# print(os.getenv("OPENXLAB_AK"))
#
# # 启动特定的脚本的方式
# # 若需要启动特定的脚本，您可在app.py 代码中通过import os的方式进行启动，例如：
#
# import os
# os.system("bash webui.sh")
# os.system("python -u launch.py")

# https://openxlab.org.cn/docs/apps/Gradio%E5%BA%94%E7%94%A8.html
import gradio as gr
import torch
import requests
from torchvision import transforms

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def predict(inp):
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences


demo = gr.Interface(fn=predict,
                    inputs=gr.inputs.Image(type="pil"),
                    outputs=gr.outputs.Label(num_top_classes=3),
                    examples=[["cheetah.jpg"]],
                    )

demo.launch()
