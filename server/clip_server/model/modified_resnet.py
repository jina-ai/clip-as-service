from open_clip.modified_resnet import ModifiedResNet


class CasModifiedResNet(ModifiedResNet):
    def forward(self, x):
        # To handle fp16 inference
        x = x.type(self.conv1.weight.dtype)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x
