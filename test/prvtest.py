model_id = "UnfilteredAI/NSFW-gen-v2"       
from diffusers import StableDiffusionPipeline                                                                                                                                                                                                 
                                                                                                                                                                                                                                              
pipe = StableDiffusionPipeline.from_pretrained(model_id)  
import torch
prompt = "girl getting fucked"       
pipe = pipe.to("cuda")       
generator = torch.Generator("cuda").manual_seed(0)
image = pipe(prompt, generator=generator).images[0]    
image.save("test.jpg")