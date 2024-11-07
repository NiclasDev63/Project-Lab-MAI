import torch


def intra_modal_consistency_loss(identity_features, I, temperature=0.1):
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
        / temperature
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
    loss /= I * T * T
    return -loss
