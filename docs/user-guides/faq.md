# FAQ

This is a list of Frequently Asked Questions about CLIP-as-service. Feel free to suggest new entries!


What is CLIP model?
: Developed by OpenAI, CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. The original CLIP Github repository [is here](https://github.com/openai/CLIP). The introduction of the CLIP model can [be found here](https://openai.com/blog/clip/).

Do I need to install `clip-server` and `clip-client` together?
: No. You can install them separately on different machines. For example, on a GPU server, you just need `clip-server`; on your laptop, you just need `clip-client`.

What is CLIP-as-service based on? The codebase seems quite small
: CLIP-as-service leverages features from [Jina](https://github.com/jina-ai/jina), which itself utilizes [DocArray](https://github.com/jina-ai/docarray). Thanks to them CLIP-as-service can be quickly built with solid infrastructure and rich features.

I had this AioRpcError, what should I do?
: If you encounter the following errors, it means you client can not connect to the server.

  ```text
     GRPCClient@28632[E]:gRPC error: StatusCode.UNAVAILABLE failed to connect to all addresses
  the ongoing request is terminated as the server is not available or closed already
  ```

  ```text
    AioRpcError: <AioRpcError of RPC that terminated with:
            status = StatusCode.UNAVAILABLE
            details = 'failed to connect to all addresses'
            debug_error_string =
    '{'created':'@1648074480.571952000','description':'Failed to pick subchannel',
    'file':'src/core/ext/filters/client_channel/client_channel.cc','file_line':312
    9,'referenced_errors':[{'created':'@1648074480.571952000','description':'faile
    d to connect to all addresses','file':'src/core/lib/transport/error_utils.cc',
    'file_line':163,'grpc_status':14}]}'
  ```

  You can try `.profile()` to {ref}`confirm it<profiling>`. If it still throws the same error, then your connection is broken.

  While it is hard to pinpoint a network problem, also out of the scope of CLIP-as-service, we here provide you a checklist that may help you to diagnose the problem: 
  - Are the IP address, port, and protocol all correct?
  - Is client and server under the same network, or a different network?
  - Is your server down?
  - Is server's port open to public?
  - Is there a firewall on the server side that restricts the port?
  - Is there a firewall on the client side that restricts the port?
  - Is the security group (on Cloud providers) correctly configured?

Why 'CLIP-as-service' why not 'CLIP-as-a-service'
: Kind of pay homage to BERT-as-service. It is not about grammatically correct anyhow.

What happened to the BERT-as-service.
: There has been no maintenance of BERT-as-service since Feb. 2019.

  CLIP-as-service is a huge upgrade of BERT-as-service, with more powerful universal embedding models that can handle both images and texts; and more solid and efficient microservice infrastructure developed in the last 2 years by Jina AI. The high-level API, especially the client side, is a drop-in replacement of the old BERT-as-service.

Where can I find the old codebase of BERT-as-service.
: In the [`bert-as-service` branch](https://github.com/jina-ai/clip-as-service/tree/bert-as-service) of the repository.