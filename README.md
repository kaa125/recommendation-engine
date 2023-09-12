# Recommendation Engine with Frequently Bought Together and Item-Based Collaborative Filtering

The Recommendation Engine with Frequently Bought Together and Item-Based Collaborative Filtering is a data-driven system designed to enhance user experiences on e-commerce platforms by providing personalized product recommendations. This project incorporates two distinct recommendation algorithms: Frequently Bought Together (FBT) and Item-Based Collaborative Filtering (IBCF). These algorithms analyze user behavior and product attributes to generate relevant product recommendations.

Key Features:

Frequently Bought Together (FBT):
Frequent Product Associations: FBT identifies products that are often bought together by users. For example, if customers frequently purchase a camera and a memory card together, the system recommends the memory card when a camera is added to the cart.
Association Strength: The system calculates the strength of associations between products using metrics like support, confidence, and lift to determine the relevance of recommendations.

Item-Based Collaborative Filtering (IBCF):
User-Item Interaction Matrix: IBCF builds a user-item interaction matrix that captures user preferences and behaviors. It tracks user interactions such as views, purchases, or ratings on products.
Similarity Calculation: The system computes item-item similarity scores (cosine similarity, Jaccard similarity, etc.) to identify products similar to those the user has shown interest in.
Personalization: IBCF provides personalized recommendations by considering a user's historical interactions and suggesting items that similar users have also interacted with.
