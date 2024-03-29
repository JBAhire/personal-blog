/**
 * Implement Gatsby's SSR (Server Side Rendering) APIs in this file.
 *
 * See: https://www.gatsbyjs.org/docs/ssr-apis/
 */

const React = require("react");
const withPrefix = require("./src/utils/withPrefix").default;

exports.onRenderBody = function({ setHeadComponents, setPostBodyComponents }) {

    setHeadComponents([
        
    ]);

    setPostBodyComponents([
        <React.Fragment>
            <script src={withPrefix('js/init.js')}/>
            <script src={withPrefix('js/page-load.js')}/>
            <script src={withPrefix('js/page-unload.js')}/>
            <script src={withPrefix('js/plugins.js')}/>
            <script src={withPrefix('js/prism.js')} data-manual/>
            
        </React.Fragment>
    ]);

};
