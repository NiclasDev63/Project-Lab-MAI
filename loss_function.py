import torch
import torch.nn as nn
import torch.nn.functional as F

class IntraModalConsistencyLoss(nn.Module):
    def __init__(self, initial_temperature=0.07):
        """
        Intra-modal consistency loss with a trainable temperature parameter.

        Parameters:
            initial_temperature (float): Initial value for the temperature parameter.
        """
        super(IntraModalConsistencyLoss, self).__init__()
        # Initialize the temperature parameter as trainable
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, identity_features):
        """
        Calculate the intra-modal consistency loss.

        Parameters:
            identity_features (torch.Tensor): Tensor of shape (N, T, d), where
                - N is the number of identities,
                - T is the number of frames per identity,
                - d is the dimensionality of identity feature vectors.

        Returns:
            torch.Tensor: The calculated loss value.
        """
        N, T, d = identity_features.shape

        # Compute pairwise similarities across time windows and identities
        # As described in the Paper: We then measure the similarity
        # between all pairs of identity vectors ⟨µi(t), µj (q)⟩, where i, j are identity indices
        # and t, q time-window indices, resulting in a T × T × N × N similarity tensor.
        identity_features = F.normalize(identity_features, p=2, dim=-1)
        similarities = (
            torch.einsum("itd,jqd->tqij", identity_features, identity_features)
            / self.temperature
        )

        # Take exponentials
        exp_similarities = torch.exp(similarities)

        # Compute numerator: similarities between same identity frames
        numerator = torch.diagonal(exp_similarities, dim1=2, dim2=3)  # Shape: (T, T, N)

        # Compute denominator: sum over all identities
        denominator = exp_similarities.sum(dim=3)  # Shape: (T, T, N)

        loss = torch.log(numerator / denominator)  # Shape: (T, T, N)

        loss = loss.sum() / (N * T * T)
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
    similarity_matrix = torch.einsum("ntd,nqd->ntq", visual_features, audio_features)

    # Extract similarity scores between audio and video
    diagonal_scores = torch.diagonal(similarity_matrix, dim1=1, dim2=2).permute(1, 0)
    similarity_matrix_exp = similarity_matrix / temperature

    # compute loss terms for similarity in both directions
    t1 = -torch.log(
        torch.exp(diagonal_scores / temperature)
        / torch.sum(torch.exp(similarity_matrix_exp), dim=-1)
    )
    t2 = -torch.log(
        torch.exp(diagonal_scores / temperature)
        / torch.sum(torch.exp(similarity_matrix_exp.transpose(1, 2)), dim=-1)
    )

    # added terms and average
    loss = (t1 + t2).mean()

    return loss
