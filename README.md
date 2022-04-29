# Boosting the Federation

This repository contains the code developed for the paper "Boosting the Federation: Cross-Silo Federated Learning without Gradient Descent".

    @inproceedings{polato2022boosting,
      author    = {Mirko Polato and
                   Roberto Esposito and Marco Aldinucci},
      title     = {Boosting the Federation: Cross-Silo Federated Learning without Gradient Descent},
      booktitle = {International Joint Conference on Neural Networks, {IJCNN} 2022, Verona,
                   Italy, 18-23 July, 2010},
      pages     = {to appear},
      publisher = {{IEEE}},
      year      = {2022},
      url       = {to appear},
      doi       = {to appear},
    }


Federated Learning has been proposed to develop better AI systems without compromising the privacy of final users and the legitimate interests of private companies. Initially deployed by Google to predict text input on mobile devices, FL has been deployed in many other industries. Since its introduction, Federated Learning mainly exploited the inner working of neural networks and other gradient descent-based algorithms by either exchanging the weights of the model or the gradients computed during learning. While this approach has been very successful, it rules out applying FL in contexts where other models are preferred, e.g., easier to interpret or known to work better.

The code in this repository implements **FL algorithms that build federated models without relying on gradient descent-based methods**. The code is complete with all the necessary to reproduce the experiments performed in the paper.




