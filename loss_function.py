import numpy as np
import torch
import torch.nn.functional as F


def intra_modal_consistency_loss(identity_features, temperature=0.1):
    """
    Calculate the intra-modal consistency loss using raw dot product

    Parameters:
        identity_features (torch.Tensor): Tensor of shape (N, T, d), where
            - N is the number of identities,
            - T is the number of frames per identity,
            - d is the dimensionality of identity feature vectors.
        temperature (float): Temperature parameter for scaling similarities.
        I: number of videos in a batch

    Returns:
        torch.Tensor: The calculated intra-modal consistency loss.
    """
    N, T, d = identity_features.shape

        # Compute pairwise similarities across time windows and identities
        # As described in the Paper: We then measure the similarity
        # between all pairs of identity vectors ⟨µi(t), µj (q)⟩, where i, j are identity indices
        # and t, q time-window indices, resulting in a T × T × N × N similarity tensor.

        similarities = (
            torch.einsum("itd,jqd->tqij", identity_features, identity_features)
            / self.temperature
        )

    # Take exponentials as per the loss formula
    exp_similarities = torch.exp(similarities)

    # Initialize the loss
    loss = 0

    # Loop over each identity and time window to compute the loss
    for t in range(T):
        q_frame_loss = 0
        for q in range(T):
            identity_loss = 0
            for i in range(N):
                # Numerator: Similarities between the same identity's frames
                # representing exp(⟨µi(t), µi (q)⟩)/ τ
                numerator = exp_similarities[t, q, i, i]

                # denominator reprensents sum(exp(⟨µi(t), µj (q)⟩)/τ)) for j in range(N)
                denominator = 0
                for j in range(N):
                    denominator = +exp_similarities[t, q, i, j]

                # Add the log term to the loss
                identity_loss += torch.log(numerator / denominator)
            q_frame_loss += identity_loss
        loss += q_frame_loss
    # Normalize the loss over all identities and time windows
    loss /= N * T * T
    return -loss

# TODO not tested yet
def cross_modal_consistency_loss(visual_features, audio_features, temperature):
    """
    Calculates cross modal consistency loss.

    visual features: Visual features tensor of shape (N, T, d)
    audio features: Audio features tensor of shape (N, T, d)
    temperature: The temperature parameter τ' to scale the similarity scores.

    """
    N, T, _ = visual_features.shape

    # Normalize features into same space, should be needed I guess even though they do not mention that in the paper
    visual_features = F.normalize(visual_features, p=2, dim=-1)
    audio_features = F.normalize(audio_features, p=2, dim=-1)

    # Compute similarity tensor
    similarity_matrix = torch.einsum('ntd,nqd->ntq', visual_features, audio_features)

    # Extract similarity scores between audio and video
    diagonal_scores = torch.diagonal(similarity_matrix, dim1=1, dim2=2).permute(1, 0)
    similarity_matrix_exp = similarity_matrix / temperature

    # compute loss terms for similarity in both directions
    t1 = -torch.log(
        torch.exp(diagonal_scores / temperature) /
        torch.sum(torch.exp(similarity_matrix_exp), dim=-1)
    )
    t2 = -torch.log(
        torch.exp(diagonal_scores / temperature) /
        torch.sum(torch.exp(similarity_matrix_exp.transpose(1, 2)), dim=-1)
    )

    # added terms and average
    loss = (t1 + t2).mean()

    return loss
