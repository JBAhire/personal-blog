const postQuery = `{
    posts: allMarkdownRemark(
        filter: { fileAbsolutePath: { regex: "/posts/" } }
    ) {
        edges {
            node {
                objectID: id
                frontmatter {
                    title
                    date(formatString: "MMM D, YYYY")
                    subtitle
                    description: excerpt
                    thumb_image
                }
                fields {
                    slug
                }
                excerpt(pruneLength: 5000)
            }
        }
    }
}`;

const flatten = (arr) =>
    arr.map(({ node: { frontmatter, ...rest } }) => ({
        ...frontmatter,
        ...rest
    }));
const settings = { attributesToSnippet: [`excerpt:20`] };

const queries = [
    {
        query: postQuery,
        transformer: ({ data }) => flatten(data.posts.edges),
        indexName: `Posts`,
        settings
    }
];

module.exports = queries;
